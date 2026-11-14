import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3, BrainCircuit,
  CheckCircle2, ChevronRight, CircleHelp, Database, FileUp, GitBranch, LayoutDashboard,
  LoaderCircle, Network, RefreshCw, ShieldAlert, ShieldCheck, Upload, Users, Waypoints
} from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const money = n => new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(n || 0)
const pct = n => `${(100 * (n || 0)).toFixed(2)}%`

function App() {
  const [page, setPage] = useState('Overview')
  const [health, setHealth] = useState({status:'checking', model_trained:false})
  const [model, setModel] = useState(null)
  const [trainFile, setTrainFile] = useState(null)
  const [trainBusy, setTrainBusy] = useState(false)
  const [trainError, setTrainError] = useState('')
  const [graph, setGraph] = useState(null)
  const [graphError, setGraphError] = useState('')
  const [graphBusy, setGraphBusy] = useState(false)
  const [graphFile, setGraphFile] = useState(null)
  const [predictFile, setPredictFile] = useState(null)
  const [predictionRows, setPredictionRows] = useState(null)
  const [predictError, setPredictError] = useState('')
  const [predictBusy, setPredictBusy] = useState(false)

  const nav = [
    {name:'Overview', icon:LayoutDashboard},
    {name:'Fraud Prediction', icon:BrainCircuit},
    {name:'Model Evaluation', icon:BarChart3},
    {name:'Graph Analysis', icon:Network},
    {name:'Project Details', icon:CircleHelp},
  ]

  async function refreshStatus() {
    try {
      const [h, m] = await Promise.all([
        fetch(`${API}/api/health`).then(r => r.json()),
        fetch(`${API}/api/model/status`).then(r => r.json())
      ])
      setHealth(h)
      if (m.trained) setModel(m)
    } catch {
      setHealth({status:'offline', model_trained:false})
    }
  }
  useEffect(() => { refreshStatus(); loadDemoGraph() }, [])

  async function trainModel(e) {
    e.preventDefault()
    if (!trainFile) { setTrainError('Choose creditcard.csv first.'); return }
    setTrainBusy(true); setTrainError('')
    try {
      const fd = new FormData(); fd.append('file', trainFile)
      const r = await fetch(`${API}/api/model/train`, {method:'POST', body:fd})
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Training failed')
      setModel(data); await refreshStatus()
    } catch (err) { setTrainError(err.message) }
    finally { setTrainBusy(false) }
  }

  async function loadDemoGraph() {
    setGraphBusy(true); setGraphError('')
    try {
      const r = await fetch(`${API}/api/graph/demo`)
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Could not load graph')
      setGraph(data)
    } catch (err) { setGraphError(err.message) }
    finally { setGraphBusy(false) }
  }

  async function analyzeGraphUpload(e) {
    e.preventDefault()
    if (!graphFile) { setGraphError('Choose a transfer CSV first.'); return }
    setGraphBusy(true); setGraphError('')
    try {
      const text = await graphFile.text()
      const lines = text.trim().split(/\r?\n/)
      if (lines.length < 2) throw new Error('CSV needs a header and at least one transfer.')
      const headers = lines[0].split(',').map(x=>x.trim().toLowerCase())
      const find = candidates => headers.findIndex(h=>candidates.includes(h))
      const si = find(['sender','source','from','sender_id'])
      const ri = find(['receiver','target','to','receiver_id'])
      const ai = find(['amount','value','transfer_amount'])
      if (si<0 || ri<0 || ai<0) throw new Error('CSV headers must include Sender, Receiver, Amount.')
      const transfers = lines.slice(1).filter(Boolean).map(line=>{
        const cols=line.split(',')
        return {sender:cols[si]?.trim(), receiver:cols[ri]?.trim(), amount:Number(cols[ai])}
      }).filter(t=>t.sender && t.receiver && Number.isFinite(t.amount) && t.amount>=0)
      const r = await fetch(`${API}/api/graph/analyze`, {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({transfers})
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Graph analysis failed')
      setGraph(data)
    } catch (err) { setGraphError(err.message) }
    finally { setGraphBusy(false) }
  }

  async function predictCsv(e) {
    e.preventDefault()
    if (!predictFile) { setPredictError('Choose a transaction CSV first.'); return }
    setPredictBusy(true); setPredictError('')
    try {
      const fd = new FormData(); fd.append('file', predictFile)
      const r = await fetch(`${API}/api/model/predict-csv`, {method:'POST', body:fd})
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Prediction failed')
      setPredictionRows(data)
    } catch (err) { setPredictError(err.message) }
    finally { setPredictBusy(false) }
  }

  const metrics = model?.metrics
  const cm = model?.confusion_matrix
  const classData = model?.dataset_info?.class_counts ? [
    {name:'Legitimate', value:model.dataset_info.class_counts.legitimate, color:'#3b8c7a'},
    {name:'Fraud', value:model.dataset_info.class_counts.fraud, color:'#d96a58'}
  ] : []

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Waypoints size={22}/></div><div><strong>FraudGraph</strong><span>AI ANALYTICS</span></div></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav>{nav.map(item => {const Icon=item.icon; return <button key={item.name} className={`nav-item ${page===item.name?'active':''}`} onClick={()=>setPage(item.name)}><Icon size={18}/><span>{item.name}</span>{page===item.name&&<ChevronRight size={15} className="nav-arrow"/>}</button>})}</nav>
      <div className="sidebar-bottom">
        <div className={`api-status ${health.status==='ok'?'online':'offline'}`}><span className="status-dot"/><span>API {health.status==='ok'?'Connected':'Not connected'}</span></div>
        <div className="sidebar-note">Academic prototype<br/>Transaction ML + graph analytics</div>
      </div>
    </aside>
    <main className="main">
      <header className="topbar"><div><span className="eyebrow">FRAUD INTELLIGENCE / WORKSPACE</span><h1>{page}</h1></div><div className="topbar-right"><div className="model-status"><span className={`status-dot ${health.model_trained?'green':'amber'}`}/>{health.model_trained?'Model trained':'Model not trained'}</div><button className="icon-button" title="Refresh API status" onClick={refreshStatus}><RefreshCw size={17}/></button></div></header>
      {health.status!=='ok' && <div className="alert error"><AlertTriangle size={18}/><div><strong>Backend not connected</strong><p>Start the FastAPI server at {API}. See README for setup steps.</p></div></div>}
      {page==='Overview' && <section className="page-content">
        <div className="welcome"><div><div className="welcome-tag">FRAUDGRAPH AI · PROJECT PROTOTYPE</div><h2>See the pattern<br/>behind the transaction.</h2><p>Analyze suspicious transactions and explore connected account activity in one workspace.</p><button className="button button-light" onClick={()=>setPage('Fraud Prediction')}>Open fraud analysis <ChevronRight size={16}/></button></div><div className="welcome-art"><div className="orbit orbit-a"/><div className="orbit orbit-b"/><div className="orbit orbit-c"/><div className="core"><ShieldCheck size={30}/></div><div className="orbit-node node-a"/><div className="orbit-node node-b"/><div className="orbit-node node-c"/><div className="orbit-node node-d"/></div></div>
        <div className="stat-grid">
          <Stat icon={Database} label="Dataset" value={metrics?money(metrics.rows):'Not loaded'} sub={metrics?`${metrics.columns} columns`:'Upload CSV to begin'}/>
          <Stat icon={ShieldAlert} label="Fraud records" value={metrics?money(metrics.fraud_train+metrics.fraud_test):'—'} sub="Known labels in dataset"/>
          <Stat icon={BrainCircuit} label="Model F1-score" value={metrics?metrics.f1.toFixed(3):'—'} sub={metrics?'Held-out test split':'Train a model first'}/>
          <Stat icon={Network} label="Graph accounts" value={graph?.summary?.node_count ?? '—'} sub={graph?`${graph.summary.edge_count} transfer links`:'Graph unavailable'}/>
        </div>
        <div className="content-grid">
          <div className="panel"><PanelTitle title="Workflow" subtitle="From labeled data to graph investigation"/><div className="workflow">{[['01','Upload dataset','Load the labeled transaction CSV.'],['02','Train baseline model','Fit and evaluate Logistic Regression.'],['03','Analyze transfer graph','Find cycles and unusual outgoing volumes.'],['04','Review flagged accounts','Inspect graph features and rule explanations.']].map(x=><div className="workflow-row" key={x[0]}><span className="step-num">{x[0]}</span><div><strong>{x[1]}</strong><p>{x[2]}</p></div><CheckCircle2 size={17} className="step-icon"/></div>)}</div></div>
          <div className="panel"><PanelTitle title="Transaction class balance" subtitle="Actual dataset counts after model training"/>{classData.length?<><div className="chart-wrap"><ResponsiveContainer width="100%" height={220}><PieChart><Pie data={classData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={86} paddingAngle={3}>{classData.map(x=><Cell key={x.name} fill={x.color}/>)}</Pie><Tooltip formatter={v=>money(v)}/><Legend/></PieChart></ResponsiveContainer></div><div className="class-foot"><span><i className="legend-dot green-dot"/>Legitimate <b>{money(classData[0].value)}</b></span><span><i className="legend-dot red-dot"/>Fraud <b>{money(classData[1].value)}</b></span></div></>:<div className="empty-state"><Database size={25}/><p>Train the model to display the class distribution.</p></div>}</div>
        </div>
      </section>}

      {page==='Fraud Prediction' && <section className="page-content">
        <div className="page-intro"><div><h2>Transaction classifier</h2><p>Train a baseline model on labeled data, then score unseen transaction rows.</p></div><span className="pill"><span className="status-dot green"/>Logistic Regression</span></div>
        <div className="panel"><PanelTitle title="1. Train model" subtitle="Upload the Kaggle credit-card dataset (creditcard.csv)"/><form className="upload-row" onSubmit={trainModel}><label className="file-drop"><FileUp size={24}/><span><strong>{trainFile?.name||'Choose CSV dataset'}</strong><small>Required columns include Time, Amount and Class</small></span><input type="file" accept=".csv" onChange={e=>setTrainFile(e.target.files[0])}/></label><button className="button primary" disabled={trainBusy}>{trainBusy?<><LoaderCircle className="spin" size={16}/>Training…</>:<><BrainCircuit size={17}/>Train model</>}</button></form>{trainError&&<div className="inline-error">{trainError}</div>}{metrics&&<div className="success-line"><CheckCircle2 size={17}/>Model trained · {money(metrics.rows)} rows · {metrics.train_rows?money(metrics.train_rows):'—'} training rows</div>}</div>
        <div className="panel"><PanelTitle title="2. Score transaction rows" subtitle="Upload a CSV with the same feature columns used during training. Class is optional and ignored."/><form className="upload-row" onSubmit={predictCsv}><label className="file-drop"><Upload size={24}/><span><strong>{predictFile?.name||'Choose transaction CSV'}</strong><small>Columns must match the training feature schema</small></span><input type="file" accept=".csv" onChange={e=>setPredictFile(e.target.files[0])}/></label><button className="button secondary" disabled={predictBusy||!health.model_trained}>{predictBusy?<><LoaderCircle className="spin" size={16}/>Scoring…</>:<>Run predictions <ChevronRight size={16}/></>}</button></form>{predictError&&<div className="inline-error">{predictError}</div>}{predictionRows&&<><div className="success-line">Scored {money(predictionRows.total_rows)} rows{predictionRows.truncated?' · first 500 displayed':''}</div><DataTable rows={predictionRows.rows} columns={['predicted_class','prediction','fraud_probability']}/><button className="button secondary" onClick={()=>downloadCsv(predictionRows.rows,'fraud_predictions.csv')}>Download displayed results CSV</button></>}</div>
        <div className="notice"><AlertTriangle size={17}/><span>The model is a baseline. A predicted fraud label is a screening signal, not proof of fraud.</span></div>
      </section>}

      {page==='Model Evaluation' && <section className="page-content">
        {!metrics?<EmptyPrompt text="Train a model on the Fraud Prediction page to view evaluation results."/>:<>
          <div className="metric-grid"><Metric label="Precision" value={metrics.precision} help="Share of predicted fraud cases that were fraud"/><Metric label="Recall" value={metrics.recall} help="Share of actual fraud cases detected"/><Metric label="F1-score" value={metrics.f1} help="Balance between precision and recall"/><Metric label="Average precision" value={metrics.average_precision} help="Area summary of precision–recall performance"/></div>
          <div className="content-grid eval-grid"><div className="panel"><PanelTitle title="Confusion matrix" subtitle="Held-out test set"/><ConfusionMatrix cm={cm}/><div className="cm-legend"><span>Rows = actual class</span><span>Columns = predicted class</span></div></div><div className="panel"><PanelTitle title="Test split summary" subtitle="Stratified 80/20 split"/><div className="summary-list">{[['Training rows',metrics.train_rows],['Testing rows',metrics.test_rows],['Fraud in training',metrics.fraud_train],['Fraud in testing',metrics.fraud_test],['Missing values',metrics.missing_values],['Duplicate rows',metrics.duplicate_rows]].map(x=><div key={x[0]}><span>{x[0]}</span><b>{money(x[1])}</b></div>)}</div><div className="notice compact"><AlertTriangle size={15}/><span>Class imbalance makes accuracy alone insufficient.</span></div></div></div>
          <div className="panel"><PanelTitle title="Classification report" subtitle="Per-class precision, recall and F1-score"/><DataTable rows={Object.entries(model.report||{}).filter(([k,v])=>typeof v==='object'&&v!==null).map(([name,v])=>({class:name,precision:v.precision,recall:v.recall,'f1-score':v['f1-score'],support:v.support}))} columns={['class','precision','recall','f1-score','support']}/></div>
        </>}
      </section>}

      {page==='Graph Analysis' && <section className="page-content">
        <div className="page-intro"><div><h2>Account relationship explorer</h2><p>Inspect directed transfers and review accounts flagged by transparent exploratory rules.</p></div><button className="button secondary" onClick={loadDemoGraph}><RefreshCw size={15}/>Load demo graph</button></div>
        <div className="panel"><PanelTitle title="Analyze transfer CSV" subtitle="Required columns: Sender, Receiver, Amount"/><form className="upload-row" onSubmit={analyzeGraphUpload}><label className="file-drop"><FileUp size={24}/><span><strong>{graphFile?.name||'Choose transfer CSV'}</strong><small>One row per transfer · account IDs are labels</small></span><input type="file" accept=".csv" onChange={e=>setGraphFile(e.target.files[0])}/></label><button className="button primary" disabled={graphBusy}>{graphBusy?<><LoaderCircle size={16} className="spin"/>Analyzing…</>:<>Analyze network <ChevronRight size={16}/></>}</button></form>{graphError&&<div className="inline-error">{graphError}</div>}</div>
        {graph&&<><div className="metric-grid graph-metrics"><Metric label="Accounts" value={graph.summary.node_count} help="Unique account IDs"/><Metric label="Transfer links" value={graph.summary.edge_count} help="Directed sender-to-receiver links"/><Metric label="Flagged accounts" value={graph.summary.flagged_count} help="Matched one or more exploratory rules"/><Metric label="Volume threshold" value={money(graph.summary.threshold)} help="75th percentile of outgoing volume"/></div><div className="panel"><PanelTitle title="Transfer network" subtitle="Red nodes are flagged; teal nodes did not meet the current rules."/><NetworkGraph graph={graph}/><div className="graph-legend"><span><i className="legend-dot red-dot"/>Flagged account</span><span><i className="legend-dot green-dot"/>Other account</span><span className="muted">Arrow direction = transfer direction</span></div></div><div className="panel"><PanelTitle title="Account-level findings" subtitle="Review why each account was flagged"/><DataTable rows={graph.accounts.map(a=>({account:a.id,incoming:a.incoming_transfers,outgoing:a.outgoing_transfers,incoming_amount:money(a.incoming_amount),outgoing_amount:money(a.outgoing_amount),status:a.flagged?'Flagged':'Not flagged',reason:a.reason}))} columns={['account','incoming','outgoing','incoming_amount','outgoing_amount','status','reason']}/><button className="button secondary" onClick={()=>downloadCsv(graph.accounts.map(a=>({account:a.id,...a})),'graph_account_summary.csv')}>Download account summary</button></div><div className="notice"><AlertTriangle size={17}/><span>{graph.note} The built-in example is synthetic; graph flags do not establish real-world wrongdoing.</span></div></>}
      </section>}

      {page==='Project Details' && <section className="page-content">
        <div className="panel prose"><PanelTitle title="About FraudGraph AI" subtitle="Project scope and methodology"/><h3>Problem</h3><p>Financial fraud can involve suspicious transactions and coordinated activity among accounts. Transaction-level classifiers may not capture account-to-account relationships.</p><h3>Transaction model</h3><p>The backend trains Logistic Regression with class weighting on the uploaded labeled dataset. It reports precision, recall, F1-score, average precision, a confusion matrix, and a classification report.</p><h3>Graph analysis</h3><p>Accounts are nodes and transfers are directed edges. The backend summarizes incoming/outgoing amounts and flags nodes involved in directed cycles or in the top quartile of outgoing transfer volume.</p><h3>Data limitation</h3><p>The public credit-card dataset has no account IDs or transfer edges. Graph analysis uses separate synthetic or user-supplied transfer data. The current graph component is not a trained GNN.</p></div>
      </section>}
      <footer className="footer"><span>FraudGraph AI · Academic prototype</span><span>Predictions and graph flags require human review</span></footer>
    </main>
  </div>
}

function Stat({icon:Icon,label,value,sub}) { return <div className="stat-card"><div className="stat-top"><span>{label}</span><Icon size={18}/></div><strong>{value}</strong><small>{sub}</small></div> }
function PanelTitle({title,subtitle}) { return <div className="panel-title"><div><h3>{title}</h3><p>{subtitle}</p></div></div> }
function Metric({label,value,help}) { return <div className="metric-card"><span>{label}</span><strong>{typeof value==='number'&&value<=1?value.toFixed(3):value}</strong><small>{help}</small></div> }
function EmptyPrompt({text}) { return <div className="panel empty-prompt"><Activity size={28}/><p>{text}</p></div> }
function DataTable({rows,columns}) { if(!rows?.length) return <p className="muted">No rows to display.</p>; return <div className="table-wrap"><table><thead><tr>{columns.map(c=><th key={c}>{c.replaceAll('_',' ')}</th>)}</tr></thead><tbody>{rows.slice(0,100).map((r,i)=><tr key={i}>{columns.map(c=><td key={c}>{typeof r[c]==='number'?(c.includes('probability')?pct(r[c]):Number.isInteger(r[c])?money(r[c]):r[c].toFixed(4)):String(r[c]??'—')}</td>)}</tr>)}</tbody></table>{rows.length>100&&<small className="muted">Showing first 100 rows.</small>}</div> }
function ConfusionMatrix({cm}) { if(!cm) return null; const labels=[['True negative',cm[0][0]],['False positive',cm[0][1]],['False negative',cm[1][0]],['True positive',cm[1][1]]]; return <div className="confusion-grid">{labels.map((x,i)=><div className={`cm-cell cm-${i}`} key={x[0]}><span>{x[0]}</span><strong>{money(x[1])}</strong></div>)}</div> }
function NetworkGraph({graph}) {
  const {nodes,edges}=graph
  const positions=useMemo(()=>{
    const out={}
    const cx=400,cy=190,rx=300,ry=135
    nodes.forEach((n,i)=>{const a=(2*Math.PI*i/nodes.length)-Math.PI/2;out[n.id]={x:cx+rx*Math.cos(a),y:cy+ry*Math.sin(a)}})
    return out
  },[nodes])
  return <div className="network-canvas"><svg viewBox="0 0 800 400" role="img" aria-label="Account transfer network">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#93a2b6"/></marker></defs>
    {edges.map((e,i)=>{const a=positions[e.source],b=positions[e.target]; if(!a||!b)return null; const dx=b.x-a.x,dy=b.y-a.y,len=Math.sqrt(dx*dx+dy*dy)||1;const sx=a.x+dx/len*20,sy=a.y+dy/len*20,tx=b.x-dx/len*23,ty=b.y-dy/len*23;return <g key={i}><line x1={sx} y1={sy} x2={tx} y2={ty} stroke="#aeb9c8" strokeWidth="1.5" markerEnd="url(#arrow)"/><text x={(sx+tx)/2} y={(sy+ty)/2-5} className="edge-label">{money(e.amount)}</text></g>})}
    {nodes.map(n=>{const p=positions[n.id];return <g key={n.id}><circle cx={p.x} cy={p.y} r="19" fill={n.flagged?'#d96a58':'#3b8c7a'} stroke="white" strokeWidth="3"/><text x={p.x} y={p.y+4} textAnchor="middle" className="node-label">{n.id}</text></g>})}
  </svg></div>
}
function downloadCsv(rows, filename) {
  if (!rows?.length) return
  const keys=Object.keys(rows[0])
  const quote=v=>`"${String(v??'').replaceAll('"','""')}"`
  const csv=[keys.map(quote).join(','),...rows.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\n')
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'})
  const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url)
}

export default App
