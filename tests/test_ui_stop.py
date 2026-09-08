import asyncio
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import pytest

from ezexl3.ui.build_locks import cleanup_locks, snapshot_locks, prepare_build_cache
from ezexl3.ui.server import Job, JobManager


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux process inspection')


@pytest.fixture(autouse=True)
def isolated_cache_and_processes(tmp_path, monkeypatch):
    # Real descriptors for this test and its children, without unrelated desktop
    # services making cleanup assertions depend on host ptrace permissions.
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', str(tmp_path))
    iterdir = Path.iterdir
    existing = {p.name for p in Path('/proc').iterdir()}
    existing.discard(str(os.getpid()))

    def test_processes(path):
        entries = iterdir(path)
        if path == Path('/proc'):
            return (p for p in entries if p.name not in existing)
        return entries

    monkeypatch.setattr(Path, 'iterdir', test_processes)


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', str(tmp_path))
    path = tmp_path / 'exllamav3_ext' / 'lock'
    path.parent.mkdir()
    return path


def test_cleanup_preserves_live_owner_and_removes_abandoned_lock(lock_path):
    with lock_path.open('w'):
        locks = snapshot_locks()
        assert cleanup_locks(locks) == []
        assert lock_path.exists()
    assert cleanup_locks(locks) == [lock_path]


def test_cleanup_preserves_replaced_lock(lock_path):
    lock_path.touch()
    locks = snapshot_locks()
    lock_path.rename(lock_path.with_name('old_lock'))
    lock_path.touch()
    assert cleanup_locks(locks) == []
    assert lock_path.exists()


def test_cleanup_skips_when_process_enumeration_denied(lock_path):
    lock_path.touch()
    locks = snapshot_locks()
    with patch.object(Path, 'iterdir', side_effect=PermissionError):
        assert cleanup_locks(locks) == []
    assert lock_path.exists()


def test_cleanup_ignores_uninspectable_helper_processes(lock_path):
    # systemd's sd-pam runs under the user's uid with an unreadable fd table in
    # every login session. It never builds extensions, so it must not veto the
    # cleanup: doing so made cleanup a no-op on real machines.
    import subprocess
    lock_path.touch()
    locks = snapshot_locks()
    helper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    iterdir = Path.iterdir

    def denied_for_helper(path):
        if path == Path('/proc') / str(helper.pid) / 'fd':
            raise PermissionError(13, 'Permission denied', str(path))
        return iterdir(path)

    try:
        with patch.object(Path, 'iterdir', denied_for_helper):
            assert cleanup_locks(locks) == [lock_path]
        assert not lock_path.exists()
    finally:
        helper.kill()
        helper.wait()


def test_startup_recovers_abandoned_lock(lock_path):
    lock_path.touch()
    original = os.environ['TORCH_EXTENSIONS_DIR']
    assert 'Removed interrupted' in prepare_build_cache()[0]
    assert not lock_path.exists()
    assert os.environ['TORCH_EXTENSIONS_DIR'] == original


@pytest.mark.parametrize('unreadable', [False, True])
def test_startup_uses_independent_cache_when_lock_cannot_be_cleared(lock_path, unreadable):
    with lock_path.open('w'):
        if unreadable:
            with patch.object(Path, 'iterdir', side_effect=PermissionError):
                messages = prepare_build_cache()
        else:
            messages = prepare_build_cache()
        assert lock_path.exists()
        cache = Path(os.environ['TORCH_EXTENSIONS_DIR'])
        assert cache.is_dir()
        assert cache != lock_path.parent.parent
        assert 'separate PyTorch build cache' in messages[-1]
        assert prepare_build_cache() == []


def test_cli_prepares_cache_before_command_import(monkeypatch):
    from ezexl3.cli import main

    class Prepared(Exception):
        pass

    def prepare():
        raise Prepared

    monkeypatch.setattr('ezexl3.ui.build_locks.prepare_build_cache', prepare)
    with pytest.raises(Prepared):
        main(['repo', '-m', '/unused', '-b', '3'])


def test_default_cache_and_unrelated_locks(tmp_path, monkeypatch):
    monkeypatch.delenv('TORCH_EXTENSIONS_DIR', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    root = tmp_path / 'torch_extensions' / 'py312_cu128'
    lock = root / 'exllamav3_ext' / 'lock'
    unrelated = root / 'another_extension' / 'lock'
    for path in (lock, unrelated):
        path.parent.mkdir(parents=True)
        path.touch()
    assert cleanup_locks(snapshot_locks()) == [lock]
    assert unrelated.exists()


async def spawn_job(script, *args):
    manager = JobManager()
    job = Job('test', [sys.executable, '-c', script, *map(str, args)])
    job.process = await asyncio.create_subprocess_exec(
        *job.cmd, start_new_session=True, stdout=asyncio.subprocess.PIPE,
    )
    job.status = 'running'
    manager.jobs[job.id] = job
    assert await asyncio.wait_for(job.process.stdout.readline(), 5) == b'ready\n'
    return manager, job


@pytest.mark.parametrize('stale', [False, True])
def test_stop_cleans_lock_before_exit_event_and_next_start(lock_path, stale):
    async def run():
        if stale:
            lock_path.touch()
        manager, job = await spawn_job(
            "import sys,time; " + ("" if stale else "f=open(sys.argv[1], 'w'); ")
            + "print('ready', flush=True); time.sleep(60)",
            lock_path,
        )
        waiter = asyncio.create_task(manager._wait_exit(job, job.process))
        original = cleanup_locks

        def checked_cleanup(locks):
            assert job.process.returncode is not None
            assert manager.live_job() is job
            assert not any(e['type'] == 'exit' for e in job.output)
            return original(locks)

        try:
            with patch('ezexl3.ui.server.cleanup_locks', side_effect=checked_cleanup):
                await manager.stop(job.id)
            await waiter
            assert not lock_path.exists()
            assert manager.live_job() is None
            assert job.status == 'stopped'
            assert job.output[-1]['type'] == 'exit'
            assert 'Removed interrupted' in job.output[-2]['text']
        finally:
            if job.process.returncode is None:
                os.killpg(job.process.pid, signal.SIGKILL)
                await job.process.wait()

    asyncio.run(run())


def test_stop_interrupts_first_so_the_builder_releases_its_own_lock(lock_path):
    async def run():
        # Mimics torch's JIT build: lock released in a finally block, which only
        # runs if the process is interrupted (SIGINT), not terminated.
        manager, job = await spawn_job(
            "import os,sys,time; f=open(sys.argv[1], 'w'); print('ready', flush=True)\n"
            "try:\n    time.sleep(60)\n"
            "finally:\n    f.close(); os.remove(sys.argv[1])",
            lock_path,
        )
        waiter = asyncio.create_task(manager._wait_exit(job, job.process))
        try:
            await manager.stop(job.id)
            await waiter
            assert job.returncode == -signal.SIGINT
            assert not lock_path.exists()
            assert not any('Removed interrupted' in e.get('text', '') for e in job.output)
        finally:
            if job.process.returncode is None:
                os.killpg(job.process.pid, signal.SIGKILL)
                await job.process.wait()

    asyncio.run(run())


def test_stop_preserves_another_builds_lock(lock_path):
    async def run():
        manager, job = await spawn_job(
            "import time; print('ready', flush=True); time.sleep(60)",
        )
        try:
            with lock_path.open('w'):
                await manager.stop(job.id)
                assert lock_path.exists()
        finally:
            if job.process.returncode is None:
                os.killpg(job.process.pid, signal.SIGKILL)
                await job.process.wait()

    asyncio.run(run())


def test_stop_kills_surviving_child_before_cleanup(lock_path):
    async def run():
        # The child retains the lock and ignores TERM; the parent exits first.
        script = """
import os, signal, sys, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    f = open(sys.argv[1], 'w')
    print('ready', flush=True)
    os.close(1)
    time.sleep(60)
else:
    time.sleep(60)
"""
        manager, job = await spawn_job(script, lock_path)
        wait_for = asyncio.wait_for

        async def shorter_timeout(awaitable, timeout):
            return await wait_for(awaitable, timeout=0.2)

        try:
            with patch('ezexl3.ui.server.asyncio.wait_for', side_effect=shorter_timeout):
                await manager.stop(job.id)
            assert not lock_path.exists()
            assert manager.live_job() is None
        finally:
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await job.process.wait()

    asyncio.run(run())
