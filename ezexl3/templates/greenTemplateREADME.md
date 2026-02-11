---
license: apache-2.0
base_model: {{AUTHOR}}/{{MODEL}}
base_model_relation: quantized
quantized_by: {{USER}}
tags:
- {{QUANT_METHOD}}
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  .dashboard-container {
    font-family: 'JetBrains Mono', monospace;
    background-color: #030503;
    background-image: 
      linear-gradient(0deg, rgba(0, 255, 65, 0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 255, 65, 0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #00ff41;
    padding: 40px;
    border: 1px solid #008f11;
    border-radius: 8px;
    /* Soft Green Halo Effect */
    box-shadow: 0 0 30px rgba(0, 143, 17, 0.15), inset 0 0 15px rgba(0, 255, 65, 0.05);
    margin: 20px;
  }

  .dashboard-header {
    margin-bottom: 35px;
  }

  .dashboard-header h1 {
    color: #00ff41;
    font-size: 1.6em;
    margin: 0;
    padding-left: 15px;
    border-left: 3px solid #00ff41;
    text-transform: uppercase;
    letter-spacing: 2px;
    /* Text Glow */
    text-shadow: 0 0 10px rgba(0, 255, 65, 0.6);
  }

  .meta-tag {
    font-size: 0.8em;
    color: #00ff41;
    background: rgba(0, 59, 0, 0.3);
    padding: 4px 12px;
    border: 1px solid #00ff41;
    border-radius: 0px;
    margin-top: 12px;
    display: inline-block;
    box-shadow: 0 0 8px rgba(0, 255, 65, 0.3);
  }

  .content-panel {
    background-color: rgba(0, 10, 0, 0.9);
    border: 1px solid #005a11;
    border-radius: 2px;
    margin-bottom: 25px;
    overflow: hidden;
    /* Border Bloom */
    box-shadow: 0 0 10px rgba(0, 90, 17, 0.2);
  }

  .panel-title {
    background-color: #001a00;
    color: #00ff41;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #008f11;
    text-shadow: 0 0 5px rgba(0, 255, 65, 0.4);
  }

  .panel-body {
    padding: 20px;
  }

  .table-wrapper {
    display: inline-block;
    border: 1px solid #004d00; 
  }

  .data-table {
    border-collapse: collapse;
    font-size: 0.85em;
    width: auto;
    margin: 0;
  }

  .data-table th {
    text-align: left;
    color: #00ff41;
    background-color: rgba(0, 30, 0, 0.8);
    padding: 12px 20px;
    border-bottom: 1px solid #008f11;
    border-right: 1px solid #004d00;
    text-shadow: 0 0 5px rgba(0, 255, 65, 0.3);
  }

  .data-table td {
    padding: 10px 20px;
    border-bottom: 1px solid #001a00;
    border-right: 1px solid #001a00;
    color: #00cc33;
  }

  .data-table tr:hover td {
    background-color: rgba(0, 255, 65, 0.05);
    color: #ffffff;
    text-shadow: 0 0 8px #00ff41;
  }

  .link-style {
    color: #00ff41;
    text-decoration: none;
    border-bottom: 1px solid #004d00;
    transition: all 0.2s ease;
  }

  .link-style:hover {
    color: #ffffff;
    border-bottom: 1px solid #00ff41;
    text-shadow: 0 0 10px #00ff41;
  }

  .terminal-box {
    background-color: #000000;
    border: 1px solid #008f11;
    padding: 18px;
    font-size: 0.85em;
    color: #00ff41;
    position: relative;
    box-shadow: inset 0 0 15px rgba(0, 255, 65, 0.1), 0 0 10px rgba(0, 143, 17, 0.1);
  }

  .terminal-box::before {
    content: "guest@matrix:~$ ";
    opacity: 0.6;
    text-shadow: none;
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
    <div class="panel-body">
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>REVISION</th>
              <th>SIZE (GiB)</th>
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