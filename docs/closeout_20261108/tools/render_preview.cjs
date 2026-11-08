/* Local-only figure rasterization and self-contained report HTML. */
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..');
const modules = process.env.GRAPHMAE2_NODE_MODULES || String.raw`C:\Users\18870\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules`;

async function main() {
  const sharp = require(path.join(modules, 'sharp'));
  const { marked } = await import(pathToFileURL(require.resolve(path.join(modules, 'marked'))).href);
  for (const stem of ['formal_B2_minus_B0', 'mechanism_S_R', 'arxiv_A_training_and_probe']) {
    const data = await fs.readFile(path.join(root, 'figures', stem + '.svg'));
    await sharp(data, { density: 180 }).png().toFile(path.join(root, 'figures', stem + '.png'));
  }
  const markdown = await fs.readFile(path.join(root, '课程报告.md'), 'utf8');
  let body = marked.parse(markdown, { gfm: true });
  body = body.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
  for (const stem of ['formal_B2_minus_B0', 'mechanism_S_R', 'arxiv_A_training_and_probe']) {
    const data = await fs.readFile(path.join(root, 'figures', stem + '.png'));
    body = body.replaceAll('figures/' + stem + '.png', 'data:image/png;base64,' + data.toString('base64'));
  }
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GraphMAE2复现与输入掩码课程改进 · 课程报告</title><style>
*{box-sizing:border-box}body{margin:0;overflow-wrap:anywhere;background:#edf1f5;color:#213246;font:17px/1.85 "Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}main{max-width:1120px;margin:44px auto;padding:56px 66px 72px;background:white;border-top:6px solid #1d638e;box-shadow:0 12px 40px #172b4010}h1{font-size:34px;line-height:1.45;margin:0 0 8px;letter-spacing:.02em}h2{font-size:25px;line-height:1.5;margin:52px 0 16px;padding-bottom:9px;border-bottom:1px solid #dce4ed}h3{font-size:20px;margin:32px 0 12px;color:#24516d}p{margin:14px 0}a{color:#17658e;text-decoration:none}a:hover{text-decoration:underline}.badge{display:inline-block;margin:0 0 25px;padding:5px 12px;background:#eef5f8;border:1px solid #c9dce8;border-radius:4px;font-size:13px;color:#315b73}.table-wrap{overflow:auto;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.65;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #dce4ed;vertical-align:top}th{background:#eef4f8;font-weight:600;color:#193c55}tbody tr:nth-child(even){background:#f7f9fb}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:22px;background:#f0f4f8;border-left:3px solid #7c9fba;font:14px/1.8 Consolas,monospace}code{font-family:Consolas,"Microsoft YaHei",monospace;font-size:.86em;background:#f0f4f8;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre code{padding:0;background:none;font-size:14px}img{display:block;width:100%;height:auto;margin:28px 0 12px;border:1px solid #e0e7ef}li{margin:9px 0}strong{color:#153c58}@media(max-width:800px){main{margin:0;padding:30px 22px;box-shadow:none}h1{font-size:28px}body{font-size:16px}table{min-width:650px}}@media print{body{background:white;font-size:10pt;line-height:1.65}main{padding:0;margin:0;max-width:none;box-shadow:none;border:0}h1{font-size:23pt}h2{font-size:16pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}table{font-size:8pt}tr,img{break-inside:avoid}a{color:inherit}.table-wrap{overflow:visible}.badge{font-size:8pt}pre,code{font-size:8pt}}
</style></head><body><main><div class="badge">内容版 · 正式结果与arxiv A已核验 · 2026-11-08</div>${body}</main></body></html>`;
  await fs.writeFile(path.join(root, '课程报告.html'), html, 'utf8');
  console.log('Rendered 3 PNG figures and self-contained report HTML.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
