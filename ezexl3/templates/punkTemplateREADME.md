---
license: apache-2.0
base_model: {{AUTHOR}}/{{MODEL}}
base_model_relation: quantized
quantized_by: {{USER}}
tags:
- {{QUANT_METHOD}}
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');

  .dashboard-container {
    font-family: 'JetBrains Mono', monospace;
    width: min(1500px, calc(100vw - 32px));
    max-width: 100%;
    margin: 0 auto;
    box-sizing: border-box;
    background-color: #0d0d0d;
    color: #00ff9f; /* Neon Green */
    padding: 40px 24px;
    border: 1px solid #333;
    border-radius: 4px;
  }

  .dashboard-header {
    margin-bottom: 35px;
    border-bottom: 2px solid #ff00ff; /* Magenta accent */
    padding-bottom: 15px;
  }

  .dashboard-header h1 {
    color: #ffffff;
    font-size: 1.6em;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 0 0 8px #ffffff, 0 0 16px #cccccc;
  }

  .meta-tag {
    font-size: 0.85em;
    color: #00ff9f;
    opacity: 0.8;
    margin-top: 12px;
    display: inline-block;
  }

  .content-panel {
    background-color: #111;
    border: 1px solid #00ff9f;
    margin-bottom: 25px;
    box-shadow: 5px 5px 0px #ff00ff; /* Hard magenta shadow */
    overflow: hidden;
  }

  .panel-title {
    background-color: #00ff9f;
    color: #0d0d0d;
    padding: 8px 20px;
    font-weight: 800;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .panel-body {
    padding: 20px;
    color: #e0e0e0;
  }

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
    border: 1px solid rgba(0, 255, 159, 0.3);
  }

  .table-wrapper {
    display: inline-block;
    margin: 0 auto;
    border: 1px solid #333;
    border-radius: 4px;
    overflow: hidden;
    max-width: calc(100% - (var(--edge-gap) * 2));
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
    color: #ff00ff;
    background-color: #000;
    padding: 12px;
    border-bottom: 2px solid #ff00ff;
    text-transform: uppercase;
  }

  .data-table td {
    padding: 9px 12px;
    border-bottom: 1px dashed #333;
    color: #00ff9f;
  }

  .data-table tr:hover td {
    background-color: #1a1a1a;
    color: #ffffff;
  }

  .link-style {
    color: #00ff9f;
    text-decoration: none;
    font-weight: 700;
    border-bottom: 1px dotted #00ff9f;
  }

  .link-style:hover {
    background-color: #ff00ff;
    color: #0d0d0d;
    border-bottom: none;
  }

  .terminal-box {
    background-color: #000;
    border: 1px solid #333;
    border-left: 3px solid #ff00ff;
    padding: 18px;
    font-size: 0.85em;
    color: #e0e0e0;
  }
</style>

<div class="dashboard-container">

  <div class="dashboard-header">
    <h1>>> {{AUTHOR}} / {{MODEL}}</h1>
    <div class="meta-tag">[QUANTS BY :: {{USER}}]</div>
  </div>

  <div class="content-panel">
    <div class="panel-title">// Information</div>
    <div class="panel-body">
      {{QUANT_METHOD}} quantizations of <b><a class="link-style" href="{{REPOLINK}}">{{MODEL}}</a></b> via
      <b><a class="link-style" href="https://github.com/turboderp-org/exllamav3">{{QUANT_TOOL}}</a></b>.
      <br/><br/>
      Repo generated automatically with
      <a class="link-style" href="https://github.com/UnstableLlama/ezexl3">ezexl3</a>.
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">// Repo Data</div>
    <div class="panel-body repo-data-body repo-data-panel">
      <img class="repo-graph" src="{{GRAPH_FILE}}" alt="Quantization graph">
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>[Revision]</th>
              <th>[GiB]</th>
              <th>[KL Div]</th>
              <th>[PPL]</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><a class="link-style" href="{{REVISION_LINK_1}}">{{BPW_VAL_1}}</a></td>
              <td>{{GIB_1}}</td>
              <td>{{KL_1}}</td>
              <td>{{PPL_1}}</td>
            </tr>
            <tr>
              <td><a class="link-style" href="{{REPOLINK}}">bf16</a></td>
              <td>{{GIB_BASE}}</td>
              <td>0.0000</td>
              <td>{{PPL_BASE}}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">SVG Catbench</div>
    <div class="panel-body repo-data-body repo-data-panel">
      {{CATBENCH_CONTENT}}
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">// CLI Download</div>
    <div class="panel-body">
      <div class="terminal-box">
        huggingface-cli download {{USER}}/{{MODEL}}-{{QUANT_METHOD}} --revision "{{DEFAULT_REVISION}}" --local-dir ./
      </div>
    </div>
  </div>

</div>