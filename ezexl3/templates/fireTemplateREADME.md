---
license: apache-2.0
base_model: {{AUTHOR}}/{{MODEL}}
base_model_relation: quantized
quantized_by: {{USER}}
tags:
- {{QUANT_METHOD}}
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@400;900&display=swap');

  .dashboard-container {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a;
    background-image: 
      linear-gradient(rgba(255, 69, 0, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 69, 0, 0.03) 1px, transparent 1px);
    background-size: 30px 30px;
    color: #f0f0f0;
    padding: 40px;
    border: 2px solid #ff4500;
    border-radius: 4px;
    box-shadow: 0 0 20px rgba(255, 69, 0, 0.2), inset 0 0 15px rgba(255, 69, 0, 0.1);
  }

  .dashboard-header {
    margin-bottom: 35px;
  }

  .dashboard-header h1 {
    font-family: 'Orbitron', sans-serif;
    color: #ff8c00;
    font-size: 1.8em;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255, 140, 0, 0.5);
    animation: pulse 4s infinite alternate;
    border-left: 5px solid #ff4500;
    padding-left: 15px;
  }

  @keyframes pulse {
    from { text-shadow: 0 0 10px rgba(255, 140, 0, 0.5); }
    to { text-shadow: 0 0 25px rgba(255, 69, 0, 0.8), 0 0 40px rgba(255, 0, 0, 0.4); }
  }

  /* The requested diagonal banner style */
  .meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
    color: #ff4500;
    background: rgba(255, 69, 0, 0.15);
    padding: 6px 25px;
    border: 1px solid #ff4500;
    margin-top: 15px;
    display: inline-block;
    font-weight: bold;
    text-transform: uppercase;
    clip-path: polygon(8% 0, 100% 0, 92% 100%, 0 100%);
  }

  .content-panel {
    background-color: rgba(25, 25, 25, 0.95);
    border: 1px solid #333;
    margin-bottom: 25px;
    border-radius: 4px;
    overflow: hidden;
  }

  .panel-title {
    background: linear-gradient(90deg, #ff4500, #ff8c00);
    color: #000;
    padding: 8px 20px;
    font-weight: 800;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Orbitron', sans-serif;
  }

  .panel-body {
    padding: 20px;
  }

  /* Geometry contract: keep repo-data layout parity with basic template */
  .repo-data-panel {
    padding: 14px 10px;
  }

  .repo-data-body {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    width: 100%;
    --edge-gap: 8px;
  }

  .repo-graph {
    display: block;
    width: min(1440px, calc(100% - (var(--edge-gap) * 2)));
    height: auto;
    margin: 0 auto;
  }

  .table-wrapper {
    display: inline-block;
    margin: 0 auto;
    overflow: hidden;
    max-width: calc(100% - (var(--edge-gap) * 2));
    border: 1px solid #444;
    background: #000;
  }

  .data-table {
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    width: auto;
    margin: 0;
  }

  .data-table th {
    text-align: left;
    color: #ff8c00;
    background-color: #1a1a1a;
    padding: 12px 20px;
    border-bottom: 2px solid #ff4500;
  }

  .data-table td {
    padding: 10px 20px;
    border-bottom: 1px solid #222;
    color: #cbd5e0;
  }

  .data-table tr:hover td {
    background-color: rgba(255, 69, 0, 0.08);
    color: #ffffff;
  }


  .data-table tr td:last-child,
  .data-table tr th:last-child {
    border-right: none;
  }

  .data-table tr:last-child td {
    border-bottom: none;
  }

  .link-style {
    color: #ff4500;
    text-decoration: none;
    font-weight: bold;
    transition: 0.2s;
  }

  .link-style:hover {
    color: #ffca28;
    text-shadow: 0 0 5px #ff4500;
  }

  .terminal-box {
    background-color: #050505;
    border: 1px solid #333;
    border-left: 3px solid #ff4500;
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    color: #ff8c00;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
  }
</style>

<div class="dashboard-container">

  <div class="dashboard-header">
    <h1>{{AUTHOR}} / {{MODEL}}</h1>
    <div class="meta-tag">QUANTIZED BY: {{USER}}</div>
  </div>

  <div class="content-panel">
    <div class="panel-title">Information</div>
    <div class="panel-body">
      {{QUANT_METHOD}} quantizations of <b><a class="link-style" href="{{REPOLINK}}">{{MODEL}}</a></b> via
      <b><a class="link-style" href="https://github.com/turboderp-org/exllamav3">{{QUANT_TOOL}}</a></b>.
      <br/>
      repo generated automatically with
      <a class="link-style" href="https://github.com/UnstableLlama/ezexl3">ezexl3</a>.
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">Repo Data</div>
    <div class="panel-body repo-data-body repo-data-panel">
      <img class="repo-graph" src="{{GRAPH_FILE}}" alt="Quantization graph">
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>REVISION</th>
              <th>GiB</th>
              <th>KL DIV</th>
              <th>PPL</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><a class="link-style" href="#">{{BPW_VAL_1}}</a></td>
              <td>x</td>
              <td>x</td>
              <td>x</td>
            </tr>
            <tr>
              <td><a class="link-style" href="#">{{BPW_VAL_2}}</a></td>
              <td>x</td>
              <td>x</td>
              <td>x</td>
            </tr>
            <tr>
              <td><a class="link-style" href="#">{{BPW_VAL_3}}</a></td>
              <td>x</td>
              <td>x</td>
              <td>x</td>
            </tr>
            <tr>
              <td><a class="link-style" href="#">{{BPW_VAL_4}}</a></td>
              <td>x</td>
              <td>x</td>
              <td>x</td>
            </tr>
            <tr>
              <td><a class="link-style" href="#">bf16</a></td>
              <td>x</td>
              <td>0</td>
              <td>x</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">CLI Download</div>
    <div class="panel-body">
      <div class="terminal-box">
        huggingface-cli download {{USER}}/{{MODEL}}-{{QUANT_METHOD}} --revision "{{DEFAULT_REVISION}}" --local-dir ./
      </div>
    </div>
  </div>

</div>