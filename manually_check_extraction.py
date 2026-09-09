import html
import json
import os
import time
from pathlib import Path

# TODO: Move over to a no cache server setup and remove the url versioning (remove each use of BUILD_ID)
# TODO: Is it a local storage error that's making the old images display?
# TODO: Make it so that if we check an image, it adds it to a marked list and automatically marks it in any other pages. May need to think about ways to do this efficiently. To do this, we would probably want to calculate all the matches at compile time.
# TODO: Add better way to deal with cache displaying removed images (we currently do a pageshow check)
# TODO: Make it so that images are ordering by the number not alphabetically ()
# TODO: Add to a separate package.
# TODO: Decide what to do about localStorage and potential alternatives
# TODO: Create a directory of true figures
# TODO: Create a directory of just flowcharts
# TODO: Also do a similar thing for checking a new extracted image (so true image on left vs new extracted image on right)
# TODO: Add a button to download the paths of all the to keep images
# TODO: Don't allow it to add the html directory if one of them doesn't exist.
# TODO: FIgure out what's wrong with pdf.js rendering or replace it
# TODO: look into ghostscript for normalisation
ROOT = Path("./experiment_data/extraction/smoking-cessation-flowchart-extraction/")
# METHOD_DIR_NAME = "docling_imgs_quality_3"
METHOD_DIR_NAME = "true_diagrams_docling_q3_p2"
OUT_DIR = Path(f"./experiment_data/extraction_htmls/{METHOD_DIR_NAME}_check_pages")
IMG_EXTS = {".png"}
BUILD_ID = str(int(time.time()))


def find_pdf_dirs(root: Path):
    pdf_dirs = set()
    for dir_item in root.iterdir():
        if not dir_item.is_dir():
            continue
        pdf_dirs.add(dir_item)
    return sorted(pdf_dirs)


def make_check_html(
    pdf_dir: Path, total_pdfs: int, pdf_idx: int, prev_html, next_html, home_html
):
    """Create HTML for this pdf_dir. Return (out_path, num_images)."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        msg = f"No PDFs found in {pdf_dir}"
        raise ValueError(msg)

    pdf_path = pdfs[0]  # assume 1 PDF per directory
    method_dir = pdf_dir / METHOD_DIR_NAME
    if not method_dir.is_dir():
        msg = f"No {METHOD_DIR_NAME} directory in {pdf_dir}"
        raise ValueError(msg)

    images = sorted(
        p for p in method_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    out_path = OUT_DIR / "pdfs" / pdf_dir.name / f"{METHOD_DIR_NAME}_check.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths relative to the HTML file's directory (for display)
    rel_pdf = os.path.relpath(pdf_path, out_path.parent)
    rel_images_page = [os.path.relpath(p, out_path.parent) for p in images]
    # Paths relative to OUT_DIR (for global reference in localStorage + index page)
    rel_images_index = [os.path.relpath(p, OUT_DIR) for p in images]

    imgs_from_root = [ROOT / os.path.relpath(p, ROOT) for p in images]

    # Navigation links (prev/next)
    if prev_html is not None:
        prev_rel = os.path.relpath(prev_html, out_path.parent)
        prev_url = f"{prev_rel}?v={BUILD_ID}"
        prev_link_html = (
            f'<a href="{html.escape(prev_url)}" id="prev-link">&larr; Prev</a>'
        )
    else:
        prev_link_html = '<span class="disabled">&larr; Prev</span>'

    if next_html is not None:
        next_rel = os.path.relpath(next_html, out_path.parent)
        next_url = f"{next_rel}?v={BUILD_ID}"
        next_link_html = (
            f'<a href="{html.escape(next_url)}" id="next-link">Next &rarr;</a>'
        )
    else:
        next_link_html = '<span class="disabled">Next &rarr;</span>'

    # Home link to the index page
    rel_home = os.path.relpath(home_html, out_path.parent)
    home_url = f"{rel_home}?v={BUILD_ID}"
    home_link_html = f'<a href="{html.escape(home_url)}">Home</a>'

    title = f"{pdf_path.name} - {METHOD_DIR_NAME}"
    escaped_title = html.escape(title)
    escaped_pdf_name = html.escape(pdf_path.name)
    escaped_method_name = html.escape(METHOD_DIR_NAME)

    thumbs_html = []
    for rel_img_page, rel_img_index, img_path, img_from_root in zip(
        rel_images_page, rel_images_index, images, imgs_from_root, strict=True
    ):
        fname = html.escape(img_path.name)
        img_id = html.escape(rel_img_index)
        thumbs_html.append(
            f"""
            <div class="thumb">
              <div class="thumb-label">{fname}</div>
              <img src="{html.escape(rel_img_page)}" loading="lazy" draggable="false">
              <div class="mark-remove-row">
                <label>
                  <input type="checkbox"
                         class="mark-remove"
                         data-img-id="{img_id}"
                         data-img-src="{html.escape(rel_img_index)}"
                         data-img-src-from-root="{html.escape(img_from_root.as_posix())}"
                         data-img-name="{fname}"
                         data-pdf-title="{escaped_pdf_name}">
                  Mark for removal
                </label>
              </div>
            </div>
            """
        )

    thumbs_block = (
        "\n".join(thumbs_html)
        if thumbs_html
        else f"<p>No images found in {escaped_method_name} directory.</p>"
    )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 8px 12px;
      border-bottom: 1px solid #ccc;
      background: #f5f5f5;
    }}
    .header-left {{
      display: flex;
      align-items: baseline;
      gap: 12px;
    }}
    .home-link a {{
      text-decoration: none;
      color: #0366d6;
      font-size: 14px;
    }}
    .home-link a:hover {{
      text-decoration: underline;
    }}
    .header-text h1 {{
      margin: 0;
      font-size: 18px;
    }}
    .header-text h2 {{
      margin: 2px 0 0 0;
      font-size: 14px;
      font-weight: normal;
      color: #555;
    }}
    .nav-buttons a {{
      text-decoration: none;
      color: #0366d6;
      margin-left: 8px;
    }}
    .nav-buttons .disabled {{
      color: #999;
      margin-left: 8px;
    }}
    .container {{
      display: flex;
      height: calc(100vh - 100px);
    }}
    .pdf-pane {{
      flex: 2;
      border-right: 1px solid #ccc;
      overflow-y: auto;
      padding: 8px;
    }}
    .pdf-grid {{
      display: grid;
      grid-auto-rows: auto;
      gap: 8px;
    }}
    .pdf-page-canvas {{
      width: 100%;
      height: auto;
      border: 1px solid #ccc;
      box-sizing: border-box;
      background-color: #fff;
    }}
    .img-pane {{
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }}
    .img-pane h2 {{
      margin-top: 0;
      font-size: 16px;
    }}
    .thumb {{
      margin-bottom: 12px;
      border-bottom: 1px solid #eee;
      padding-bottom: 8px;
      cursor: pointer;
    }}
    .thumb-label {{
      font-size: 12px;
      color: #555;
      margin-bottom: 4px;
      word-break: break-all;
    }}
    .thumb img {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    .mark-remove-row {{
      margin-top: 4px;
      font-size: 12px;
      color: #333;
    }}
    .thumb.marked {{
      background-color: #fff3f3;
    }}
    .footer-bar {{
      border-top: 1px solid #ccc;
      padding: 4px 12px;
      font-size: 12px;
      background: #f5f5f5;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .footer-left {{
      flex: 2;
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: flex-end;
    }}
    .footer-right {{
      flex: 1;
      /* Reserved for future controls */
    }}
    .footer-left label {{
      font-weight: 500;
    }}
    .footer-left select {{
      font-size: 12px;
      padding: 2px 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <div class="home-link">
        {home_link_html}
      </div>
      <div class="header-text">
        <h1>{escaped_title}</h1>
        <h2>{pdf_idx} / {total_pdfs}</h2>
      </div>
    </div>
    <div class="nav-buttons">
      {prev_link_html}
      {next_link_html}
    </div>
  </div>
  <div class="container">
    <div class="pdf-pane">
      <div id="pdf-grid" class="pdf-grid"></div>
    </div>
    <div class="img-pane">
      <h2>{len(images)} imgs</h2>
      {thumbs_block}
    </div>
  </div>
  <div class="footer-bar">
    <div class="footer-left">
      <label for="cols-select">Pages per row:</label>
      <select id="cols-select">
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4">4</option>
        <option value="5">5</option>
        <option value="6">6</option>
        <option value="7">7</option>
        <option value="8">8</option>
        <option value="9">9</option>
        <option value="10">10</option>
      </select>
    </div>
    <div class="footer-right">
      <!-- Reserved for future controls -->
    </div>
  </div>
  <script type="module">
    import * as pdfjsLib from 'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/build/pdf.mjs';

    (function() {{
      const STORAGE_KEY = 'markedToRemoveByMethod';
      const METHOD_KEY = '{METHOD_DIR_NAME}';
      const COLS_STORAGE_KEY = 'pdfPagesPerRow';
      const DEFAULT_COLS = 3;
      const pdfUrl = "{html.escape(rel_pdf)}";
      const QUALITY_SCALE = 1.5;  // Extra sharpness factor

      // --- Image marking ---

      function loadAll() {{
        try {{
          return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        }} catch (e) {{
          return {{}};
        }}
      }}

      function loadList() {{
        const all = loadAll();
        const list = all[METHOD_KEY];
        return Array.isArray(list) ? list : [];
      }}

      function saveList(list) {{
        const all = loadAll();
        all[METHOD_KEY] = list;
        localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
      }}

      let current = loadList();
      const byId = Object.create(null);
      current.forEach(item => {{ byId[item.id] = item; }});

      document.querySelectorAll('input.mark-remove').forEach(cb => {{
        const id = cb.dataset.imgId;
        const thumb = cb.closest('.thumb');

        if (byId[id]) {{
          cb.checked = true;
          thumb.classList.add('marked');
        }}

        cb.addEventListener('change', () => {{
          let list = loadList();
          if (cb.checked) {{
            const item = {{
              id: id,
              src: cb.dataset.imgSrc,
              name: cb.dataset.imgName,
              pdf: cb.dataset.pdfTitle,
              srcFromRoot: cb.dataset.imgSrcFromRoot,
            }};
            if (!list.some(x => x.id === id)) {{
              list.push(item);
            }}
            thumb.classList.add('marked');
          }} else {{
            list = list.filter(x => x.id !== id);
            thumb.classList.remove('marked');
          }}
          saveList(list);
        }});
      }});

      // Make the entire thumb area clickable to toggle the checkbox
      document.querySelectorAll('.thumb').forEach(thumb => {{
        thumb.addEventListener('click', (event) => {{
          // If the click was on the checkbox or its label area,
          // let the native behaviour handle it to avoid double toggling.
          if (event.target.closest('.mark-remove-row')) {{
            return;
          }}

          const cb = thumb.querySelector('input.mark-remove');
          if (!cb) return;

          cb.checked = !cb.checked;
          // Reuse the existing change handler to update state + localStorage
          cb.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }});
      }});

      // Keyboard navigation
      document.addEventListener('keydown', function(e) {{
        if (e.key === 'ArrowLeft') {{
          const prev = document.getElementById('prev-link');
          if (prev) {{
            window.location.href = prev.href;
          }}
        }} else if (e.key === 'ArrowRight') {{
          const next = document.getElementById('next-link');
          if (next) {{
            window.location.href = next.href;
          }}
        }}
      }});

      // If the page is restored from the back/forward cache, force a reload
      window.addEventListener('pageshow', (event) => {{
        if (event.persisted) {{
          window.location.reload();
        }}
      }});

      // --- PDF rendering with pdf.js (ESM) ---

      const pdfGrid = document.getElementById('pdf-grid');
      const colsSelect = document.getElementById('cols-select');
      let pdfDoc = null;

      function loadCols() {{
        const raw = localStorage.getItem(COLS_STORAGE_KEY);
        if (!raw) return DEFAULT_COLS;
        const n = parseInt(raw, 10);
        if (!Number.isFinite(n) || n < 1 || n > 10) return DEFAULT_COLS;
        return n;
      }}

      let currentCols = loadCols();

      function updateColsUI() {{
        if (!colsSelect) return;
        colsSelect.value = String(currentCols);
      }}

      function setGridColumns() {{
        if (!pdfGrid) return;
        pdfGrid.style.gridTemplateColumns = `repeat(${{currentCols}}, minmax(0, 1fr))`;
      }}

      function renderAllPages() {{
        if (!pdfDoc || !pdfGrid) return;

        pdfGrid.innerHTML = '';
        setGridColumns();

        const deviceScale = window.devicePixelRatio || 1;
        const outputScale = deviceScale * QUALITY_SCALE;
        const numPages = pdfDoc.numPages;

        for (let pageNum = 1; pageNum <= numPages; pageNum++) {{
          pdfDoc.getPage(pageNum).then(page => {{
            // Unscaled viewport to read intrinsic page size
            const unscaledViewport = page.getViewport({{ scale: 1.0 }});

            const gridWidth = pdfGrid.clientWidth || 800;
            const desiredCssWidth = gridWidth / currentCols - 12; // small gap

            // CSS scale so the page fits this column
            const cssScale = desiredCssWidth / unscaledViewport.width;
            const viewport = page.getViewport({{ scale: cssScale }});

            const canvas = document.createElement('canvas');
            canvas.className = 'pdf-page-canvas';
            const ctx = canvas.getContext('2d');

            // Internal pixel size (HiDPI)
            canvas.width = Math.floor(viewport.width * outputScale);
            canvas.height = Math.floor(viewport.height * outputScale);

            // CSS (pixel) size (layout size)
            canvas.style.width = Math.floor(viewport.width) + 'px';
            canvas.style.height = Math.floor(viewport.height) + 'px';

            pdfGrid.appendChild(canvas);

            const transform = outputScale !== 1
              ? [outputScale, 0, 0, outputScale, 0, 0]
              : null;

            const renderContext = {{
              canvasContext: ctx,
              transform: transform,
              viewport: viewport,
            }};
            page.render(renderContext);
          }}).catch(err => {{
            console.error('Failed to render page', pageNum, err);
          }});
        }}
      }}

      if (pdfGrid) {{
        pdfjsLib.GlobalWorkerOptions.workerSrc =
          'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/build/pdf.worker.mjs';

        pdfjsLib.getDocument({{
          url: pdfUrl,
          wasmUrl:
            'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/wasm/',
          standardFontDataUrl:
            'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/standard_fonts/',
          cMapUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/cmaps/',
          cMapPacked: true,
          iccUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/iccs/',
          standardFontDataUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.449/standard_fonts/',
        }}).promise.then(pdf => {{
          pdfDoc = pdf;
          updateColsUI();
          renderAllPages();
        }}).catch(err => {{
          console.error(err);
          pdfGrid.innerHTML = '<p>Failed to load PDF pages.</p>';
        }});
      }}

      if (colsSelect) {{
        colsSelect.addEventListener('change', function() {{
          const n = parseInt(this.value, 10);
          if (!Number.isFinite(n) || n < 1 || n > 10) return;
          currentCols = n;
          localStorage.setItem(COLS_STORAGE_KEY, String(n));
          renderAllPages();
        }});
      }}

      // Re-render on resize so pages reflow to new column width
      window.addEventListener('resize', () => {{
        renderAllPages();
      }});
    }})();
  </script>
</body>
</html>
"""  # noqa: S608

    out_path.write_text(html_text, encoding="utf-8")
    return out_path, len(images)


def main():
    pdf_dirs = find_pdf_dirs(ROOT)
    if not pdf_dirs:
        print("No PDF directories found.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / f"index_{METHOD_DIR_NAME}_check.html"

    # Precompute HTML paths in alphabetical order
    pdf_html_paths = [
        OUT_DIR / "pdfs" / d.name / f"{METHOD_DIR_NAME}_check.html" for d in pdf_dirs
    ]

    total_images = 0
    all_images_for_index = []

    if len(pdf_dirs) != len(pdf_html_paths):
        msg = "Mismatch between number of pdf dirs and html paths"
        raise ValueError(msg)

    for idx, d in enumerate(pdf_dirs):
        # Collect all images for the "unmarked images" page
        method_dir = d / METHOD_DIR_NAME
        if not method_dir.is_dir():
            msg = f"No {METHOD_DIR_NAME} directory in {d}"
            raise ValueError(msg)
        images = sorted(
            p
            for p in method_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        for img_path in images:
            rel_from_out = os.path.relpath(img_path, OUT_DIR)
            img_from_root = ROOT / os.path.relpath(img_path, ROOT)
            all_images_for_index.append(
                {
                    "id": rel_from_out,
                    "src": rel_from_out,
                    "name": img_path.name,
                    "pdf": d.name,
                    "srcFromRoot": img_from_root.as_posix(),
                }
            )

        prev_html = pdf_html_paths[idx - 1] if idx > 0 else None
        next_html = pdf_html_paths[idx + 1] if idx < len(pdf_dirs) - 1 else None

        _, num_images = make_check_html(
            d,
            len(pdf_dirs),
            idx + 1,
            prev_html,
            next_html,
            index_path,
        )
        total_images += num_images

    # Make a top-level index for convenience
    if pdf_html_paths:
        index_lines = [
            "<!doctype html>",
            "<html>",
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{html.escape(METHOD_DIR_NAME)} visual checks</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; padding: 16px; }",
            "    .layout { display: flex; gap: 24px; }",
            "    .left { flex: 1; }",
            "    .left-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }",
            "    .left-header-main h1 { margin: 0; }",
            "    .left-header-main h2 { margin: 4px 0 0 0; font-size: 14px; font-weight: normal; color: #555; }",
            "    .left-header-actions a { font-size: 14px; text-decoration: none; color: #0366d6; padding: 4px 8px; border: 1px solid #0366d6; border-radius: 4px; }",
            "    .left-header-actions a:hover { background: #0366d6; color: #fff; }",
            "    .right { flex: 1; max-height: 80vh; overflow-y: auto; border-left: 1px solid #ddd; padding-left: 16px; }",
            "    ul { list-style: none; padding-left: 0; }",
            "    li { margin-bottom: 4px; }",
            "    a { text-decoration: none; color: #0366d6; }",
            "    a:hover { text-decoration: underline; }",
            "    .marked-thumb { margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px; }",
            "    .marked-thumb img { max-width: 100%; height: auto; display: block; }",
            "    .marked-label { font-size: 12px; color: #555; margin-bottom: 4px; }",
            "    .actions-right { margin-bottom: 12px; }",
            "    button#clear-marked { font-size: 14px; padding: 4px 8px; cursor: pointer; }",
            "    button#download-marked { font-size: 14px; padding: 4px 8px; cursor: pointer; }",
            "    button#clear-all-marked { font-size: 14px; padding: 4px 8px; cursor: pointer; background-color: #d9534f; color: #fff; border: 1px solid #000; margin-left: 8px; }",
            "  </style>",
            "</head>",
            "<body>",
            '  <div class="layout">',
            '    <div class="left">',
            '      <div class="left-header">',
            '        <div class="left-header-main">',
            f"          <h1>{html.escape(METHOD_DIR_NAME)}</h1>",
            f"          <h2>{len(pdf_dirs)} pdfs,   {total_images} imgs</h2>",
            "        </div>",
            '        <div class="left-header-actions">',
            f'          <a href="unmarked_{html.escape(METHOD_DIR_NAME)}.html?v={BUILD_ID}" id="unmarked-link">Unmarked images</a>',
            "        </div>",
            "      </div>",
            "      <ul>",
        ]

        for pdf_html_path in pdf_html_paths:
            rel = os.path.relpath(pdf_html_path, OUT_DIR)
            rel_w_version = f"{rel}?v={BUILD_ID}"
            pdf_name = pdf_html_path.parent.name  # the PDF directory name
            index_lines.append(
                f'        <li><a href="{html.escape(rel_w_version)}">{html.escape(pdf_name)}</a></li>'
            )

        index_lines += [
            "      </ul>",
            "    </div>",
            '    <div class="right">',
            "      <h2>Marked to remove</h2>",
            '      <div class="actions-right">',
            '        <button id="clear-marked">Reset marked images</button>',
            '        <button id="download-marked">Download marked paths</button>',
            '        <button id="clear-all-marked">Delete all marked (all methods)</button>',
            "      </div>",
            '      <div id="marked-images"></div>',
            "    </div>",
            "  </div>",
            "  <script>",
            "    (function() {",
            "      const STORAGE_KEY = 'markedToRemoveByMethod';",
            f"      const METHOD_KEY = '{METHOD_DIR_NAME}';",
            "      function loadAll() {",
            "        try {",
            "          return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');",
            "        } catch (e) {",
            "          return {};",
            "        }",
            "      }",
            "      function loadList() {",
            "        const all = loadAll();",
            "        const list = all[METHOD_KEY];",
            "        return Array.isArray(list) ? list : [];",
            "      }",
            "      function saveListForMethod(list) {",
            "        const all = loadAll();",
            "        all[METHOD_KEY] = list;",
            "        localStorage.setItem(STORAGE_KEY, JSON.stringify(all));",
            "      }",
            "      const container = document.getElementById('marked-images');",
            "      const clearBtn = document.getElementById('clear-marked');",
            "      const downloadBtn = document.getElementById('download-marked');",
            "      const clearAllBtn = document.getElementById('clear-all-marked');",
            "      if (!container) return;",
            "      function render() {",
            "        const list = loadList();",
            "        if (!list.length) {",
            "          container.innerHTML = '<p>No images marked for removal yet.</p>';",
            "          return;",
            "        }",
            "        container.innerHTML = list.map(item => `"
            '          <div class="marked-thumb">'
            "            <div class=\"marked-label\">${item.pdf || ''} - ${item.name}</div>"
            '            <img src="${item.src}" loading="lazy">'
            "          </div>"
            "        `).join('');",
            "      }",
            "      render();",
            "      clearBtn.addEventListener('click', () => {",
            "        saveListForMethod([]);",
            "        render();",
            "      });",
            "      downloadBtn.addEventListener('click', () => {",
            "        const list = loadList();",
            "        const lines = list.map(item => item.srcFromRoot).join('\\n');",
            "        const blob = new Blob([lines], { type: 'text/plain' });",
            "        const url = URL.createObjectURL(blob);",
            "        const a = document.createElement('a');",
            "        a.href = url;",
            "        a.download = 'marked_paths.txt';",
            "        document.body.appendChild(a);",
            "        a.click();",
            "        a.remove();",
            "        URL.revokeObjectURL(url);",
            "      });",
            "      clearAllBtn.addEventListener('click', () => {",
            "        if (!confirm('Are you sure you want to delete all marked images for ALL methods?')) return;",
            "        localStorage.removeItem(STORAGE_KEY);",
            "        render();",
            "      });",
            "    })();",
            "  </script>",
            "</body>",
            "</html>",
        ]

        index_path.write_text("\n".join(index_lines), encoding="utf-8")
        print(f"Created index: {index_path}")

        unmarked_path = OUT_DIR / f"unmarked_{METHOD_DIR_NAME}.html"
        # slight safety: avoid </script> appearing in JSON
        all_images_json = json.dumps(all_images_for_index).replace("</", "<\\/")

        unmarked_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Unmarked images - {html.escape(METHOD_DIR_NAME)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }}
    h1 {{
      margin: 0;
    }}
    .header {{
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 8px 16px;
      background: #fff;
      border-bottom: 1px solid #ddd;
    }}
    .header-left {{
      display: flex;
      align-items: baseline;
      gap: 12px;
    }}
    .home-link a {{
      font-size: 14px;
      text-decoration: none;
      color: #0366d6;
      padding: 4px 8px;
      border: 1px solid #0366d6;
      border-radius: 4px;
    }}
    .home-link a:hover {{
      background: #0366d6;
      color: #fff;
    }}
    .header-right label {{
      font-size: 14px;
      font-weight: 500;
    }}
    .header-right select {{
      font-size: 14px;
      padding: 2px 4px;
    }}
    .images-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding: 16px;
    }}
    .thumb {{
      border: 1px solid #eee;
      padding: 8px;
      box-sizing: border-box;
    }}
    .thumb img {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    .thumb-label {{
      font-size: 12px;
      color: #555;
      margin-top: 4px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <div class="home-link">
        <a href="index_{METHOD_DIR_NAME}_check.html?v={BUILD_ID}">Home</a>
      </div>
      <h1>Unmarked images - {html.escape(METHOD_DIR_NAME)}</h1>
    </div>
    <div class="header-right">
      <label for="cols-select">Images per row:</label>
      <select id="cols-select">
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4" selected>4</option>
        <option value="5">5</option>
        <option value="6">6</option>
        <option value="7">7</option>
        <option value="8">8</option>
        <option value="9">9</option>
        <option value="10">10</option>
      </select>
    </div>
  </div>
  <div id="unmarked-images" class="images-grid"></div>
  <script>
    (function() {{
      const STORAGE_KEY = 'markedToRemoveByMethod';
      const METHOD_KEY = '{METHOD_DIR_NAME}';
      const ALL_IMAGES = {all_images_json};
      const COLS_STORAGE_KEY = 'unmarkedImagesPerRow';
      const DEFAULT_COLS = 4;

      function loadAll() {{
        try {{
          return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        }} catch (e) {{
          return {{}};
        }}
      }}

      function loadList() {{
        const all = loadAll();
        const list = all[METHOD_KEY];
        return Array.isArray(list) ? list : [];
      }}

      function loadCols() {{
        const raw = localStorage.getItem(COLS_STORAGE_KEY);
        if (!raw) return DEFAULT_COLS;
        const n = parseInt(raw, 10);
        if (!Number.isFinite(n) || n < 1 || n > 10) return DEFAULT_COLS;
        return n;
      }}

      const container = document.getElementById('unmarked-images');
      const colsSelect = document.getElementById('cols-select');
      let currentCols = loadCols();

      function setGridColumns() {{
        if (!container) return;
        container.style.gridTemplateColumns = `repeat(${{currentCols}}, minmax(0, 1fr))`;
      }}

      function updateColsUI() {{
        if (!colsSelect) return;
        colsSelect.value = String(currentCols);
      }}

      function render() {{
        const marked = new Set(loadList().map(item => item.id));
        const unmarked = ALL_IMAGES.filter(img => !marked.has(img.id));

        if (!unmarked.length) {{
          container.innerHTML = '<p>All images are currently marked for removal.</p>';
          return;
        }}

        setGridColumns();

        container.innerHTML = unmarked.map(item => `
          <div class="thumb">
            <img src="${{item.src}}" loading="lazy">
            <div class="thumb-label">${{item.pdf}} — ${{item.name}}</div>
          </div>
        `).join('');
      }}

      if (colsSelect) {{
        colsSelect.addEventListener('change', function() {{
          const n = parseInt(this.value, 10);
          if (!Number.isFinite(n) || n < 1 || n > 10) return;
          currentCols = n;
          localStorage.setItem(COLS_STORAGE_KEY, String(n));
          render();
        }});
      }}

      window.addEventListener('resize', () => {{
        setGridColumns();
      }});

      updateColsUI();
      render();
    }})();
  </script>
</body>
</html>
"""
        unmarked_path.write_text(unmarked_html, encoding="utf-8")
        print(f"Created unmarked images page: {unmarked_path}")
    else:
        print(
            "No check pages generated. Did you set ROOT and METHOD_DIR_NAME correctly?"
        )


if __name__ == "__main__":
    main()
