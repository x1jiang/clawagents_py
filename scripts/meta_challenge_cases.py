# BENCHMARK_PRIVATE_ORACLE
"""Frozen synthetic coding challenges with out-of-workspace deterministic graders."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap


def code(text: str) -> str:
    return textwrap.dedent(text).lstrip()


@dataclass(frozen=True)
class Case:
    prompt: str
    files: dict[str, str]
    oracle: str
    reference: dict[str, str]


LEDGER_REFERENCE = code("""
    from decimal import Decimal, InvalidOperation

    def summarize(rows):
        seen, groups = set(), {}
        for row in rows:
            event_id = row['event_id']
            if event_id in seen:
                continue
            seen.add(event_id)
            if row['status'] != 'posted':
                continue
            try:
                amount = Decimal(str(row['amount']))
            except (InvalidOperation, ValueError, TypeError):
                raise ValueError('invalid amount')
            if not amount.is_finite() or amount < 0 or amount != amount.quantize(Decimal('0.01')):
                raise ValueError('invalid amount')
            if row['kind'] not in ('charge', 'refund'):
                raise ValueError('invalid kind')
            key = (row['currency'], row['customer'])
            total, count = groups.get(key, (Decimal(0), 0))
            groups[key] = (total + (amount if row['kind'] == 'charge' else -amount), count + 1)
        return [dict(currency=k[0], customer=k[1], net=f'{v[0]:.2f}', count=v[1]) for k,v in sorted(groups.items())]
""")
LEDGER_CLI = code("""
    import json, sys
    from .report import summarize
    if __name__ == '__main__':
        with open(sys.argv[1], encoding='utf-8') as handle:
            print(json.dumps(summarize(json.load(handle))))
""")
LEDGER_ORACLE = code("""
    import json, random, subprocess, sys, tempfile
    from pathlib import Path
    from decimal import Decimal
    from ledger.report import summarize
    rng=random.Random(917)
    rows=[]; expected={}; seen=set()
    for i in range(220):
        row=dict(event_id=str(i),customer=rng.choice(['A','B','C']),currency=rng.choice(['USD','EUR']),amount=f'{rng.randrange(10000)/100:.2f}',status=rng.choice(['posted','pending']),kind=rng.choice(['charge','refund']))
        rows.append(row)
        if row['status']=='posted':
            key=(row['currency'],row['customer'])
            total,n=expected.get(key,(Decimal(0),0))
            expected[key]=(total+Decimal(row['amount'])*(1 if row['kind']=='charge' else -1),n+1)
        if i%4==0:
            rows.append(dict(row,amount='9999.99',status='posted'))
    wanted=[dict(currency=k[0],customer=k[1],net=f'{v[0]:.2f}',count=v[1]) for k,v in sorted(expected.items())]
    assert summarize(rows)==wanted
    assert summarize([])==[]
    first=dict(event_id='dup',customer='A',currency='USD',amount='1.00',status='pending',kind='charge')
    assert summarize([first,dict(first,status='posted')])==[]
    for value in ['NaN','Infinity','-0.01','0.001','not money']:
        bad=dict(first,event_id='bad',status='posted',amount=value)
        try: summarize([bad])
        except ValueError: pass
        else: raise AssertionError('invalid money accepted')
    bad=dict(first,event_id='bad',status='posted',kind='transfer')
    try: summarize([bad])
    except ValueError: pass
    else: raise AssertionError('invalid kind accepted')
    Path('oracle_input.json').write_text(json.dumps(rows))
    result=subprocess.run([sys.executable,'-m','ledger','oracle_input.json'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and json.loads(result.stdout)==wanted
    print(json.dumps({'checks':10,'passed':True}))
""")

DAG_REFERENCE = code("""
    def plan(graph, targets=None):
        deps={k:set(v) for k,v in graph.items()}
        if any(d not in deps for values in deps.values() for d in values):
            raise ValueError('unknown dependency')
        pending={k:set(v) for k,v in deps.items()}
        while pending:
            ready={k for k,v in pending.items() if not v}
            if not ready: raise ValueError('cycle')
            pending={k:v-ready for k,v in pending.items() if k not in ready}
        if targets is None:
            selected=set(deps)
        else:
            selected=set()
            def visit(k):
                if k not in deps: raise ValueError('unknown target')
                if k in selected: return
                selected.add(k)
                for d in deps[k]: visit(d)
            for k in targets: visit(k)
        remaining={k:deps[k]&selected for k in selected}
        waves=[]
        while remaining:
            ready=sorted(k for k,v in remaining.items() if not v)
            waves.append(ready)
            remaining={k:v-set(ready) for k,v in remaining.items() if k not in ready}
        return waves
""")
DAG_ORACLE = code("""
    import copy, json, random
    from planner.graph import plan
    rng=random.Random(810)
    def expected(graph,targets):
        chosen=set(graph) if targets is None else set(targets)
        while True:
            expanded=chosen|{d for k in chosen for d in graph[k]}
            if expanded==chosen: break
            chosen=expanded
        completed=set(); result=[]
        while chosen-completed:
            batch=sorted(k for k in chosen-completed if set(graph[k])<=completed)
            assert batch
            result.append(batch);completed.update(batch)
        return result
    checks=0
    for n in range(1,36):
        names=[f'n{i:02d}' for i in range(n)]
        graph={k:[d for d in names[:i] if rng.random()<0.2] for i,k in enumerate(names)}
        for targets in [None,[],rng.sample(names,min(3,n))]:
            before=copy.deepcopy(graph)
            assert plan(graph,targets)==expected(graph,targets)
            assert graph==before
            checks+=2
    assert plan({})==[]
    assert plan({'a':[],'b':['a','a']})==[['a'],['b']]
    for graph,targets in [({'a':['x']},None),({'a':['a']},None),({'a':['b'],'b':['a']},None),({'a':[]},['missing']),({'ok':[],'x':['x']},['ok'])]:
        try: plan(graph,targets)
        except ValueError: pass
        else: raise AssertionError('invalid graph accepted')
    print(json.dumps({'checks':checks+7,'passed':True}))
""")

CACHE_REFERENCE = code("""
    import time
    from collections import OrderedDict

    class TTLCache:
        def __init__(self, capacity, clock=time.monotonic):
            if capacity <= 0: raise ValueError('capacity')
            self.capacity=capacity; self.clock=clock; self.data=OrderedDict()
        def _purge(self):
            now=self.clock()
            for k,(_,deadline) in list(self.data.items()):
                if deadline is not None and now>=deadline: del self.data[k]
        def put(self,key,value,ttl=None):
            self._purge()
            self.data.pop(key,None)
            if ttl is not None and ttl<=0: return
            self.data[key]=(value,None if ttl is None else self.clock()+ttl)
            while len(self.data)>self.capacity: self.data.popitem(last=False)
        def get(self,key,default=None):
            self._purge()
            if key not in self.data: return default
            value,_=self.data[key];self.data.move_to_end(key)
            return value
        def __len__(self):
            self._purge();return len(self.data)
""")
CACHE_ORACLE = code("""
    import json, random
    from collections import OrderedDict
    from cachelib.core import TTLCache
    checks=0
    for capacity in [1,2,5,9]:
        now=[0.0]; cache=TTLCache(capacity,clock=lambda:now[0]); reference=OrderedDict()
        rng=random.Random(990+capacity)
        for _ in range(250):
            now[0]+=rng.choice([0,0.5,1,3])
            for key,(_,deadline) in list(reference.items()):
                if deadline is not None and now[0]>=deadline:del reference[key]
            key=rng.randrange(12)
            if rng.random()<0.6:
                ttl=rng.choice([None,-1,0,0.5,1,4]); value=rng.choice([None,False,0,'data',key])
                cache.put(key,value,ttl)
                reference.pop(key,None)
                if ttl is None or ttl>0:
                    reference[key]=(value,None if ttl is None else now[0]+ttl)
                    while len(reference)>capacity:reference.popitem(last=False)
            else:
                got=cache.get(key,'MISSING')
                wanted=reference[key][0] if key in reference else 'MISSING'
                assert got==wanted
                if key in reference:reference.move_to_end(key)
                checks+=1
            assert len(cache)==len(reference)
            checks+=1
    for capacity in [0,-1]:
        try:TTLCache(capacity)
        except ValueError:pass
        else:raise AssertionError('invalid capacity accepted')
    print(json.dumps({'checks':checks+2,'passed':True}))
""")

MIGRATION_REFERENCE = code("""
    def migrate(conn):
        exists=conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
        if exists and conn.execute('SELECT version FROM schema_version').fetchone()==(2,):return
        rows=conn.execute('SELECT id,email,display_name FROM users ORDER BY id').fetchall()
        keep={}
        for ident,email,name in rows:
            if email is None or not email.strip():raise ValueError('blank email')
            normalized=email.strip().lower()
            keep.setdefault(normalized,(ident,normalized,name))
        with conn:
            conn.execute('BEGIN')
            conn.execute('CREATE TABLE users_new (id INTEGER PRIMARY KEY, email TEXT NOT NULL COLLATE NOCASE UNIQUE, display_name TEXT)')
            conn.executemany('INSERT INTO users_new VALUES (?,?,?)',keep.values())
            conn.execute('DROP TABLE users')
            conn.execute('ALTER TABLE users_new RENAME TO users')
            conn.execute('CREATE TABLE schema_version (version INTEGER NOT NULL)')
            conn.execute('INSERT INTO schema_version VALUES (2)')
""")
MIGRATION_ORACLE = code("""
    import json, sqlite3
    from migrate import migrate
    def connect(rows):
        c=sqlite3.connect(':memory:')
        c.execute('CREATE TABLE users(id INTEGER PRIMARY KEY,email TEXT,display_name TEXT)')
        c.executemany('INSERT INTO users VALUES(?,?,?)',rows);c.commit();return c
    c=connect([(8,' ADA@example.com ','later'),(3,'ada@EXAMPLE.com','first'),(9,'lin@example.com','Lin')])
    migrate(c)
    expected=[(3,'ada@example.com','first'),(9,'lin@example.com','Lin')]
    assert c.execute('SELECT * FROM users ORDER BY id').fetchall()==expected
    assert c.execute('SELECT version FROM schema_version').fetchall()==[(2,)]
    before=list(c.iterdump());migrate(c);assert list(c.iterdump())==before
    try:c.execute("INSERT INTO users VALUES(10,'ADA@example.com','duplicate')")
    except sqlite3.IntegrityError:c.rollback()
    else:raise AssertionError('case-insensitive uniqueness missing')
    for bad in ['', '  ',None]:
        c=connect([(1,'good@example.com','Good'),(2,bad,'Bad')]);before=list(c.iterdump())
        try:migrate(c)
        except ValueError:pass
        else:raise AssertionError('blank email accepted')
        assert list(c.iterdump())==before
    c=connect([]);migrate(c);assert c.execute('SELECT COUNT(*) FROM users').fetchone()==(0,)
    # Failure after table creation must not leave a partially applied migration.
    c=connect([(1,'a@example.com','A')]);before=list(c.iterdump())
    import sqlite3 as sql
    c.set_authorizer(lambda action,arg1,arg2,db,trigger: sql.SQLITE_DENY if action==sql.SQLITE_DROP_TABLE and arg1=='users' else sql.SQLITE_OK)
    try:migrate(c)
    except sqlite3.DatabaseError:pass
    else:raise AssertionError('injected SQL failure did not fire')
    c.set_authorizer(None)
    assert list(c.iterdump())==before
    print(json.dumps({'checks':12,'passed':True}))
""")

CASES = {
    "ledger": Case(
        prompt="Implement the financial-event report described in SPEC.md across the ledger package. Preserve its public API and CLI. Read the existing code, add meaningful tests, and run them. Do not install dependencies.",
        files={
            "SPEC.md": "Implement ledger.report.summarize(rows) and python -m ledger INPUT.json. Each row has event_id, customer, currency, amount (decimal string), status, and kind. First occurrence of each event_id wins, even if pending. Ignore non-posted events. Posted charge adds amount, refund subtracts. Reject other kinds and invalid, negative, non-finite, or fractional-cent amounts with ValueError. Use exact decimal arithmetic. Return a list sorted by (currency,customer), each dict containing currency, customer, net (two-decimal string), count (unique posted events). Preserve zero/negative totals; never mix currencies. CLI prints only JSON. Empty input yields [].\n",
            "ledger/__init__.py": "",
            "ledger/domain.py": 'CURRENCIES = ("USD", "EUR")\n',
            "ledger/report.py": 'from .domain import CURRENCIES\n\ndef summarize(rows):\n    return [{"net": str(sum(float(r["amount"]) for r in rows))}]\n',
            "ledger/__main__.py": "from .report import summarize\nprint(summarize([]))\n",
            "test_public.py": "import unittest\nfrom ledger.report import summarize\nclass Tests(unittest.TestCase):\n    def test_empty(self): self.assertEqual(summarize([]), [])\n",
        },
        oracle=LEDGER_ORACLE,
        reference={
            "ledger/report.py": LEDGER_REFERENCE,
            "ledger/__main__.py": LEDGER_CLI,
        },
    ),
    "dependency_planner": Case(
        prompt="Repair planner.graph.plan according to SPEC.md. Handle target closure, deterministic parallel waves, invalid dependencies and cycles. Keep input data unchanged. Add and run tests. No dependencies.",
        files={
            "SPEC.md": "plan(graph, targets=None) accepts a dict mapping task names to lists of prerequisite names. Return execution waves: each wave is the lexicographically sorted list of all remaining tasks whose prerequisites completed in earlier waves. None selects all tasks; [] selects none; other targets select those tasks and their transitive prerequisites. Validate the ENTIRE graph for unknown dependencies and cycles even if they are outside target closure; invalid graphs or unknown targets raise ValueError. Duplicate dependency names are harmless. Empty graph returns []. Do not mutate graph or targets.\n",
            "planner/__init__.py": "",
            "planner/graph.py": "def plan(graph, targets=None):\n    return [list(graph)]\n",
            "planner/cli.py": "import json, sys\nfrom .graph import plan\ndef main(): print(json.dumps(plan(json.load(sys.stdin))))\n",
            "test_public.py": 'import unittest\nfrom planner.graph import plan\nclass Tests(unittest.TestCase):\n    def test_chain(self): self.assertEqual(plan({"b":["a"],"a":[]}), [["a"],["b"]])\n',
        },
        oracle=DAG_ORACLE,
        reference={"planner/graph.py": DAG_REFERENCE},
    ),
    "ttl_cache": Case(
        prompt="Implement cachelib.core.TTLCache to the contract in SPEC.md. Preserve injected-clock support, cover expiry and LRU interaction with tests, and run your tests. No dependencies.",
        files={
            "SPEC.md": "TTLCache(capacity, clock=time.monotonic) implements put(key,value,ttl=None), get(key,default=None), and len(cache). capacity<=0 raises ValueError. ttl=None never expires; ttl<=0 removes an existing key without storing a replacement. Other deadlines are relative to the injected clock; expiration occurs at now>=deadline. Successful get promotes the key to MRU; misses do not. put/update promotes to MRU and resets TTL. Remove expired entries before evicting the least recently used live entry. len reports only live entries. Support None, False, and zero values. Do not sleep or use the real clock when one was injected.\n",
            "cachelib/__init__.py": "",
            "cachelib/core.py": "import time\nclass TTLCache:\n    def __init__(self,capacity,clock=time.monotonic): self.data={}\n    def put(self,key,value,ttl=None): self.data[key]=value\n    def get(self,key,default=None): return self.data.get(key,default)\n    def __len__(self): return len(self.data)\n",
            "test_public.py": 'import unittest\nfrom cachelib.core import TTLCache\nclass Tests(unittest.TestCase):\n    def test_put_get(self):\n        c=TTLCache(2); c.put("a",1); self.assertEqual(c.get("a"),1)\n',
        },
        oracle=CACHE_ORACLE,
        reference={"cachelib/core.py": CACHE_REFERENCE},
    ),
    "sqlite_migration": Case(
        prompt="Implement migrate.migrate(conn) according to SPEC.md. Preserve data, make migration idempotent and atomic even on SQL failures, add tests and run them. Use sqlite3 from the standard library only.",
        files={
            "SPEC.md": "Input is a sqlite3 connection with no active transaction, containing users(id INTEGER PRIMARY KEY,email TEXT,display_name TEXT). migrate(conn) upgrades it in place: strip/lower emails; deduplicate normalized email retaining the smallest id and that row's display_name. Reject NULL/blank email with ValueError and leave database entirely unchanged. Result users keeps those three columns and enforces NOT NULL and case-insensitive UNIQUE email. Create schema_version(version INTEGER NOT NULL) with exactly one row 2. Calling again at version 2 changes nothing. Empty users works. Any SQL failure must propagate and roll back the ENTIRE migration, including created tables/schema changes; no temporary table may remain. Caller already committed legacy data.\n",
            "migrate.py": 'def migrate(conn):\n    conn.execute("UPDATE users SET email=lower(email)")\n    conn.commit()\n',
            "legacy.sql": 'CREATE TABLE users(id INTEGER PRIMARY KEY,email TEXT,display_name TEXT);\nINSERT INTO users VALUES(1," Ada@example.com ","Ada");\n',
            "test_public.py": 'import sqlite3, unittest\nfrom migrate import migrate\nclass Tests(unittest.TestCase):\n    def test_empty(self):\n        c=sqlite3.connect(":memory:"); c.execute("CREATE TABLE users(id INTEGER PRIMARY KEY,email TEXT,display_name TEXT)"); c.commit(); migrate(c)\n        self.assertEqual(c.execute("SELECT COUNT(*) FROM users").fetchone()[0],0)\n',
        },
        oracle=MIGRATION_ORACLE,
        reference={"migrate.py": MIGRATION_REFERENCE},
    ),
}
