---
license: apache-2.0
base_model: {{AUTHOR}}/{{MODEL}}
base_model_relation: quantized
quantized_by: {{USER}}
tags:
- {{QUANT_METHOD}}
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');

  .dashboard-container {
    font-family: 'Inter', sans-serif;
    background-color: #1a1b1e;
    background-image: radial-gradient(#2d2f34 1px, transparent 1px);
    background-size: 20px 20px;
    color: #e0e0e0;
    padding: 40px;
    border: 1px solid #4a4d53;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
  }

  .dashboard-header {
    margin-bottom: 35px;
  }

  .dashboard-header h1 {
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
    font-size: 1.6em;
    margin: 0;
    padding-left: 15px;
    border-left: 4px solid #4dabf7;
  }

  .meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
    color: #4dabf7;
    background: rgba(77, 171, 247, 0.1);
    padding: 4px 12px;
    border: 1px solid rgba(77, 171, 247, 0.3);
    border-radius: 4px;
    margin-top: 12px;
    display: inline-block;
  }

  .content-panel {
    background-color: #25262b;
    border: 1px solid #4a4d53;
    border-radius: 8px;
    margin-bottom: 25px;
    overflow: hidden;
  }

  .panel-title {
    background-color: #4dabf7;
    color: #000000;
    padding: 8px 20px;
    font-weight: 800;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .panel-body {
    padding: 20px;
  }

  .table-wrapper {
    display: inline-block;
    border: 1px solid #666a73; 
    border-radius: 4px;
    overflow: hidden;
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
    color: #ffffff;
    background-color: #2d2f34;
    padding: 12px 20px;
    border-bottom: 2px solid #666a73;
    border-right: 1px solid #4a4d53;
  }

  .data-table td {
    padding: 10px 20px;
    border-bottom: 1px solid #4a4d53;
    border-right: 1px solid #4a4d53;
  }

  .data-table tr td:last-child, 
  .data-table tr th:last-child {
    border-right: none;
  }

  .data-table tr:last-child td {
    border-bottom: none;
  }

  .data-table tr:hover td {
    background-color: rgba(77, 171, 247, 0.05);
  }

  .link-style {
    color: #4dabf7;
    text-decoration: none;
  }

  .link-style:hover {
    text-decoration: underline;
    color: #ffffff;
  }

  .terminal-box {
    background-color: #0c0d0e;
    border: 1px solid #4a4d53;
    border-radius: 6px;
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    color: #cbd5e0;
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
    <div class="panel-title">Quantization Matrix</div>
    <div class="panel-body">
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>REVISION</th>
              <th>SIZE (GiB)</th>
              <th>K/L DIV</th>
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
    <div class="panel-title">CLI Access</div>
    <div class="panel-body">
      <div class="terminal-box">
        huggingface-cli download {{USER}}/{{MODEL}}-{{QUANT_METHOD}} --revision "{{DEFAULT_REVISION}}" --local-dir ./
      </div>
    </div>
  </div>

</div>