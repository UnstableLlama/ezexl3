---
license: apache-2.0
base_model: {{AUTHOR}}/{{MODEL}}
base_model_relation: quantized
quantized_by: {{USER}}
tags:
- {{QUANT_METHOD}}
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Crimson+Pro:ital,wght@0,400;0,900;1,400&family=Inter:wght@400;600&display=swap');

  .dashboard-container {
    font-family: 'Inter', sans-serif;
    background-color: #0a0d0a; /* Very dark forest floor */
    /* Native CSS Texture: Simulates organic grit/bark */
    background-image: 
      repeating-linear-gradient(45deg, rgba(20, 30, 20, 0.2) 0px, rgba(20, 30, 20, 0.2) 1px, transparent 1px, transparent 2px),
      linear-gradient(to bottom, rgba(45, 34, 25, 0.4), transparent);
    color: #b8c2b8;
    padding: 40px;
    border: 1px solid #1b241b;
    border-radius: 2px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.7);
  }

  .dashboard-header {
    margin-bottom: 35px;
    border-left: 4px solid #3e4d3e;
    padding-left: 20px;
  }

  .dashboard-header h1 {
    font-family: 'Crimson Pro', serif;
    color: #4b634b; /* Deep moss green */
    font-size: 2.2em;
    font-weight: 900;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 2px 2px 0px rgba(0,0,0,0.5);
  }

  .meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: #d4c4a8; /* Parchment */
    background: #3d2b1f; /* Cedar Brown */
    padding: 6px 20px;
    margin-top: 15px;
    display: inline-block;
    font-weight: bold;
    border: 1px solid #4a3728;
    /* Organic "Bark" shape */
    clip-path: polygon(2% 0%, 98% 5%, 100% 100%, 0% 95%);
  }

  .content-panel {
    background-color: rgba(18, 22, 18, 0.95);
    border: 1px solid #252a25;
    margin-bottom: 25px;
    /* Deep shadow to feel "heavy" */
    box-shadow: 8px 8px 0px rgba(10, 15, 10, 0.5);
  }

  .panel-title {
    background: #1b241b; /* Dark Moss */
    color: #6b826b;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2d362d;
  }

  .panel-body {
    padding: 25px;
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
    border: 1px solid #2d362d;
    background: #0d110d;
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
    color: #4b634b;
    background-color: #151a15;
    padding: 15px 25px;
    border-bottom: 2px solid #3d2b1f; /* Brown accent line */
  }

  .data-table td {
    padding: 12px 25px;
    border-bottom: 1px solid #1a201a;
    color: #8fa38f;
  }

  .data-table tr:hover td {
    background-color: rgba(61, 43, 31, 0.15); /* Cedar hover */
    color: #d4c4a8;
  }


  .data-table tr td:last-child,
  .data-table tr th:last-child {
    border-right: none;
  }

  .data-table tr:last-child td {
    border-bottom: none;
  }

  .link-style {
    color: #6b826b;
    text-decoration: none;
    font-weight: bold;
    border-bottom: 1px solid #3d2b1f;
  }

  .link-style:hover {
    color: #d4c4a8;
    border-bottom: 1px solid #d4c4a8;
  }

  .terminal-box {
    background-color: #070907;
    border: 1px solid #1b241b;
    border-top: 2px solid #3d2b1f;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    color: #5a6e5a;
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
      {{QUANT_METHOD}} quantizations of <b><a class="link-style" href="{{REPOLINK}}">{{MODEL}}</a></b> via <b>{{QUANT_TOOL}}</b>.
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