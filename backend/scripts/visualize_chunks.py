"""Render a parsed PDF as a self-contained HTML page showing how it was chunked.

Each section chunk's spans are drawn over the page image (colour = topic, thick border =
emphasised chunk). Click a chunk in the sidebar to jump to it; hover a box to see its breadcrumb.

    uv run python scripts/visualize_chunks.py ../data/private/note.pdf ../data/private/viz.html

Output may embed private material, so keep it under data/private/ (gitignored).
"""

import base64
import html
import json
import sys
import uuid
from pathlib import Path

import pymupdf

from app.core.config import DomainConfig, get_settings, load_domain_config
from app.ingestion.chunker import chunk_pages
from app.ingestion.furniture import extract_metadata
from app.ingestion.parser import PyMuPDFParser

SCALE = 1.5
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def build(pdf: Path, out: Path) -> None:
    config: DomainConfig = load_domain_config(get_settings().config_path)
    pages = PyMuPDFParser(config.parsing).parse(pdf)
    chunks = chunk_pages(pages, str(uuid.uuid4()), pdf.stem, config.chunking)
    topic_colour = {
        c.id: PALETTE[i % len(PALETTE)]
        for i, c in enumerate(c for c in chunks if c.kind == "topic")
    }

    page_imgs: list[dict[str, object]] = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE))
            data = base64.b64encode(pix.tobytes("jpeg", jpg_quality=72)).decode()
            page_imgs.append({"w": page.mediabox.width, "h": page.mediabox.height, "img": data})

    sections = [c for c in chunks if c.kind == "section"]
    boxes = []
    for index, c in enumerate(sections):
        for sp in c.spans:
            boxes.append({"chunk": index, "page": sp.page - 1, "bbox": sp.bbox})
    side = [
        {
            "i": i,
            "crumb": c.breadcrumb,
            "chars": c.char_count,
            "emphasis": round(c.emphasis, 2),
            "emphasised": c.emphasised,
            "colour": topic_colour[c.parent_id or ""],
            "page": c.page,
            "preview": c.body[:140],
        }
        for i, c in enumerate(sections)
    ]
    n_topics = len(topic_colour)
    stats = (
        f"{len(pages)} pages · {n_topics} topics · {len(sections)} sections · "
        f"{sum(c.emphasised for c in sections)} emphasised · metadata {extract_metadata(pages)}"
    )
    payload = json.dumps({"pages": page_imgs, "boxes": boxes, "side": side}, ensure_ascii=False)
    out.write_text(TEMPLATE.replace("__STATS__", html.escape(stats)).replace("__DATA__", payload))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {stats}")


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chunk viewer</title>
<style>
:root{--bg:#f6f7f9;--fg:#111;--card:#fff;--line:#d9dde3;--mut:#667}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e8e8;--card:#1d2025;--line:#333842;--mut:#9aa}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
header b{font-size:15px}header span{color:var(--mut);margin-left:8px}
main{display:grid;grid-template-columns:340px 1fr;gap:0;height:calc(100vh - 46px)}
aside{overflow:auto;border-right:1px solid var(--line);padding:8px}
#pages{overflow:auto;padding:16px;display:flex;flex-direction:column;align-items:center;gap:18px}
.item{background:var(--card);border:1px solid var(--line);border-left:6px solid;border-radius:6px;
 padding:6px 8px;margin:0 0 6px;cursor:pointer}
.item.on{outline:2px solid currentColor}.item small{color:var(--mut)}
.item .crumb{font-weight:600;font-size:13px}.star{color:#d97706;font-weight:700}
.page{position:relative;box-shadow:0 1px 6px #0003;background:#fff;max-width:100%}
.page img{display:block;width:100%}.pn{position:absolute;top:4px;right:8px;font-size:11px;color:#888}
.box{position:absolute;opacity:.28;border-radius:2px;transition:opacity .15s}
.box.emph{outline:2px solid #d97706}.box.on{opacity:.65}
@media(max-width:800px){main{grid-template-columns:1fr;grid-template-rows:40vh 1fr}}
</style></head><body>
<header><b>Chunk viewer</b><span>__STATS__</span>
<span>colour = topic · orange outline = emphasised chunk · click a chunk to jump</span></header>
<main><aside id="side"></aside><div id="pages"></div></main>
<script>
const D=__DATA__;const side=document.getElementById('side'),pg=document.getElementById('pages');
const els=D.pages.map((p,i)=>{const d=document.createElement('div');d.className='page';
 d.style.width=Math.round(p.w*1.4)+'px';
 d.innerHTML='<img src="data:image/jpeg;base64,'+p.img+'"><span class="pn">p.'+(i+1)+'</span>';
 pg.appendChild(d);return d});
const boxEls=D.side.map(()=>[]);
D.boxes.forEach(b=>{const p=D.pages[b.page],s=D.side[b.chunk],[x0,y0,x1,y1]=b.bbox;
 const e=document.createElement('div');e.className='box'+(s.emphasised?' emph':'');
 e.style.cssText='left:'+x0/p.w*100+'%;top:'+(p.h-y1)/p.h*100+'%;width:'+(x1-x0)/p.w*100+'%;height:'+(y1-y0)/p.h*100+'%;background:'+s.colour;
 e.title=s.crumb+' ('+s.chars+' chars, emphasis '+Math.round(s.emphasis*100)+'%)';
 e.onclick=()=>pick(b.chunk,false);els[b.page].appendChild(e);boxEls[b.chunk].push(e)});
const items=D.side.map(s=>{const d=document.createElement('div');d.className='item';d.style.borderLeftColor=s.colour;
 d.innerHTML='<div class="crumb">'+(s.emphasised?'<span class="star">★ </span>':'')+s.crumb.replace(/</g,'&lt;')+'</div><small>p.'+s.page+' · '+s.chars+' chars · emphasis '+Math.round(s.emphasis*100)+'%</small>';
 d.onclick=()=>pick(s.i,true);side.appendChild(d);return d});
let cur=-1;function pick(i,scroll){if(cur>=0){items[cur].classList.remove('on');boxEls[cur].forEach(e=>e.classList.remove('on'))}
 cur=i;items[i].classList.add('on');boxEls[i].forEach(e=>e.classList.add('on'));
 if(scroll&&boxEls[i][0])boxEls[i][0].scrollIntoView({block:'center',behavior:'smooth'});
 else items[i].scrollIntoView({block:'nearest'})}
</script></body></html>"""

if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
