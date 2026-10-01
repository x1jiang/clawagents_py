const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourceRoot = path.join(__dirname, '../examples/hip_dislocation/web/static');

function client(fetch) {
  const downloads = [];
  class Element {
    constructor(tag = 'div', value = '') {
      this.tag = tag;
      this.value = value;
      this.children = [];
      this.style = {};
      this.className = '';
      this.classList = {toggle() {}};
      this.files = [];
      this._text = '';
    }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
    append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
    replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
    setAttribute() {}
    click() {
      if (this.tag === 'a') downloads.push({
        href: this.href,
        filename: this.download,
        attached: Boolean(this.parent),
        rowsAtDownload: elements.get('rows').children.length,
      });
    }
    remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
  }
  const html = fs.readFileSync(path.join(sourceRoot, 'index.html'), 'utf8');
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  elements.get('filter').value = 'all';
  const files = [];
  const context = vm.createContext({
    document: {
      body: new Element('body'),
      getElementById: id => elements.get(id),
      createElement: tag => new Element(tag),
      querySelectorAll: selector => {
        const found = [];
        function visit(node) {
          if (selector === '.original' && node.className === 'original') found.push(node);
          node.children.forEach(visit);
        }
        for (const node of elements.values()) visit(node);
        return found;
      },
    },
    Option: class extends Element {
      constructor(text, value) { super('option', value); this.textContent = text; }
    },
    window: {addEventListener() {}},
    URL: {createObjectURL(blob) { files.push(blob); return 'blob:export'; }, revokeObjectURL() {}},
    Blob, TextDecoder, AbortController, fetch,
    setTimeout: callback => callback(),
  });
  vm.runInContext(fs.readFileSync(path.join(sourceRoot, 'app.js'), 'utf8'), context);
  return {context, elements, files, downloads, evaluate: code => vm.runInContext(code, context)};
}

const prediction = {label: 'posterior', needs_review: false, obturator_screen: 'unlikely', evidence: 'posterior', reason: 'Posterior displacement.'};
function stream() {
  const payload = Buffer.from([
    {event: 'result', index: 0, id: '7', prediction, seconds: 1},
    {event: 'complete'},
  ].map(e => JSON.stringify(e)).join('\n') + '\n');
  let read = false;
  return {ok: true, body: {getReader: () => ({async read() {
    if (read) return {done: true};
    read = true;
    return {done: false, value: payload};
  }})}};
}

test('a completed round stays reviewable until CSV export clears the session', async () => {
  const app = client(async () => stream());
  app.evaluate('switchTab("paste")');
  app.elements.get('text').value = 'SYNTHETIC ORIGINAL SOURCE: posterior displacement.';
  await app.elements.get('run').onclick();
  assert.ok(app.elements.get('text').value.includes('SYNTHETIC ORIGINAL SOURCE'));
  assert.equal(app.evaluate('results.length'), 1);
  assert.ok(app.elements.get('rows').textContent.includes('SYNTHETIC ORIGINAL SOURCE'));
  assert.ok(app.elements.get('rows').textContent.includes('posterior'));
  app.elements.get('download').onclick();
  assert.equal(app.elements.get('text').value, '');
  assert.equal(app.evaluate('input.length + results.length'), 0);
  assert.equal(app.elements.get('rows').children.length, 0);
  assert.equal(app.elements.get('run').disabled, true);
});

test('Excel/CSV export retains labels then clears inputs, mappings and results', async () => {
  const app = client(async url => url.startsWith('/api/parse')
    ? {ok: true, json: async () => ({sheets: [{name: 'Reports', headers: ['ID', 'Report', 'Reference'], rows: [['7', 'SYNTHETIC ORIGINAL SOURCE posterior', 'posterior']]}]})}
    : stream());
  app.elements.get('file').files = [{name: 'reports.csv', size: 100}];
  await app.elements.get('file').onchange();
  app.elements.get('sheet').value = '0';
  app.elements.get('report-column').value = '1';
  app.elements.get('id-column').value = '0';
  app.elements.get('reference-column').value = '2';
  await app.elements.get('run').onclick();
  assert.ok(app.elements.get('performance').textContent.includes('100%'));
  app.elements.get('download').onclick();
  assert.equal(app.downloads.length, 1);
  assert.equal(app.downloads[0].attached, true);
  assert.ok(app.downloads[0].rowsAtDownload > 0);
  assert.match(app.downloads[0].filename, /^hip-report-predictions-.*\.csv$/);
  const csv = await app.files[0].text();
  assert.ok(csv.includes('"7","posterior"'));
  assert.ok(!csv.includes('SYNTHETIC ORIGINAL SOURCE'));
  assert.equal(app.evaluate('input.length + results.length + sheets.length'), 0);
  assert.equal(app.elements.get('file').value, '');
  assert.equal(app.elements.get('sheet').children.length, 0);
  assert.equal(app.elements.get('rows').children.length, 0);
  assert.equal(app.elements.get('performance').children.length, 0);
  assert.equal(app.elements.get('dashboard').hidden, true);
  assert.equal(app.elements.get('download').disabled, true);
});

test('failed and cancelled rounds remain retryable and manual clearing releases inputs', async () => {
  for (const name of ['Error', 'AbortError']) {
    const app = client(async () => { const error = new Error('Unavailable'); error.name = name; throw error; });
    app.evaluate('switchTab("paste")');
    app.elements.get('text').value = 'SYNTHETIC SOURCE';
    await app.elements.get('run').onclick();
    assert.equal(app.elements.get('text').value, 'SYNTHETIC SOURCE');
    assert.equal(app.elements.get('run').disabled, false);
    app.elements.get('clear').onclick();
    assert.equal(app.elements.get('text').value, '');
    assert.equal(app.evaluate('input.length'), 0);
  }
});

test('a late upload response cannot repopulate a cleared session', async () => {
  let resolve;
  const pendingResponse = new Promise(r => {resolve = r;});
  const app = client(() => pendingResponse);
  app.elements.get('file').files = [{name: 'reports.csv', size: 100}];
  const upload = app.elements.get('file').onchange();
  app.elements.get('clear').onclick();
  resolve({ok: true, json: async () => ({sheets: [{name: 'Reports', headers: ['Report'], rows: [['SYNTHETIC SOURCE']]}]})});
  await upload;
  assert.equal(app.evaluate('sheets.length'), 0);
  assert.equal(app.elements.get('mapping').hidden, true);
  assert.equal(app.elements.get('sheet').children.length, 0);
});
