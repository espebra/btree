// Copyright 2014-2022 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package btree

import (
	"math/rand"
	"sort"
	"testing"
)

// kv is ordered by key only, so that replacements can be observed via val.
type kv struct {
	key, val int
}

func kvLess(a, b kv) bool { return a.key < b.key }

// model is a trivially correct reference implementation: a sorted slice.
type model []kv

func (m model) find(k int) (int, bool) {
	i := sort.Search(len(m), func(i int) bool { return m[i].key >= k })
	return i, i < len(m) && m[i].key == k
}

func (m *model) replaceOrInsert(x kv) (kv, bool) {
	i, ok := m.find(x.key)
	if ok {
		old := (*m)[i]
		(*m)[i] = x
		return old, true
	}
	*m = append(*m, kv{})
	copy((*m)[i+1:], (*m)[i:])
	(*m)[i] = x
	return kv{}, false
}

func (m *model) delete(k int) (kv, bool) {
	i, ok := m.find(k)
	if !ok {
		return kv{}, false
	}
	old := (*m)[i]
	*m = append((*m)[:i], (*m)[i+1:]...)
	return old, true
}

// bulkInsert inserts xs, which must be in strictly ascending key order, and
// returns what ReplaceOrInsert should have returned for each.  It merges in
// O(len(m)+len(xs)) rather than shifting the slice once per item.
func (m *model) bulkInsert(xs []kv) (old []kv, replaced []bool) {
	old, replaced = make([]kv, len(xs)), make([]bool, len(xs))
	out := make(model, 0, len(*m)+len(xs))
	i := 0
	for j, x := range xs {
		for i < len(*m) && (*m)[i].key < x.key {
			out = append(out, (*m)[i])
			i++
		}
		if i < len(*m) && (*m)[i].key == x.key {
			old[j], replaced[j] = (*m)[i], true
			i++
		}
		out = append(out, x)
	}
	*m = append(out, (*m)[i:]...)
	return
}

// deleteRange deletes every key in [lo, hi].
func (m *model) deleteRange(lo, hi int) {
	i, _ := m.find(lo)
	j, _ := m.find(hi + 1)
	*m = append((*m)[:i], (*m)[j:]...)
}

func (m model) clone() model { return append(model(nil), m...) }

// filter returns the items of m satisfying keep, in ascending order.
func (m model) filter(keep func(k int) bool) (out []kv) {
	for _, x := range m {
		if keep(x.key) {
			out = append(out, x)
		}
	}
	return
}

func reversed(s []kv) []kv {
	out := make([]kv, len(s))
	for i, x := range s {
		out[len(s)-1-i] = x
	}
	return out
}

// checkInvariants verifies the structural B-tree invariants of t.
func checkInvariants[T any](t *testing.T, tr *BTreeG[T]) {
	t.Helper()
	if tr.root == nil {
		if tr.length != 0 {
			t.Fatalf("nil root but length %d", tr.length)
		}
		return
	}
	less := tr.cow.less
	leafDepth := -1
	count := 0
	var walk func(n *node[T], depth int, lo, hi *T, shared bool)
	walk = func(n *node[T], depth int, lo, hi *T, shared bool) {
		// Copy-on-write only ever creates owned nodes beneath owned parents,
		// so a node not owned by this tree has no owned descendants.  Clear
		// relies on this.
		if n.cow != tr.cow {
			shared = true
		} else if shared {
			t.Fatalf("owned node beneath a shared node")
		}
		if n != tr.root && len(n.items) < tr.minItems() {
			t.Fatalf("node underflow: %d items < %d", len(n.items), tr.minItems())
		}
		if len(n.items) > tr.maxItems() {
			t.Fatalf("node overflow: %d items > %d", len(n.items), tr.maxItems())
		}
		if len(n.children) != 0 && len(n.children) != len(n.items)+1 {
			t.Fatalf("bad child count %d for %d items", len(n.children), len(n.items))
		}
		for i := range n.items {
			if i > 0 && !less(n.items[i-1], n.items[i]) {
				t.Fatalf("items out of order")
			}
			if lo != nil && !less(*lo, n.items[i]) {
				t.Fatalf("item below lower bound")
			}
			if hi != nil && !less(n.items[i], *hi) {
				t.Fatalf("item above upper bound")
			}
		}
		count += len(n.items)
		if len(n.children) == 0 {
			if leafDepth == -1 {
				leafDepth = depth
			} else if leafDepth != depth {
				t.Fatalf("leaves at differing depths %d and %d", leafDepth, depth)
			}
			return
		}
		for i, c := range n.children {
			clo, chi := lo, hi
			if i > 0 {
				clo = &n.items[i-1]
			}
			if i < len(n.items) {
				chi = &n.items[i]
			}
			walk(c, depth+1, clo, chi, shared)
		}
	}
	walk(tr.root, 0, nil, nil, false)
	if count != tr.length {
		t.Fatalf("length %d but counted %d items", tr.length, count)
	}
}

func collect(f func(ItemIteratorG[kv])) (out []kv) {
	f(func(x kv) bool {
		out = append(out, x)
		return true
	})
	return
}

// collectN collects at most n items, testing early termination.
func collectN(n int, f func(ItemIteratorG[kv])) (out []kv) {
	f(func(x kv) bool {
		if len(out) >= n {
			return false
		}
		out = append(out, x)
		return true
	})
	return
}

func eqKV(a, b []kv) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func checkQueries(t *testing.T, tr *BTreeG[kv], m model, a, b, limit int) {
	t.Helper()
	ka, kb := kv{key: a}, kv{key: b}
	check := func(name string, got, want []kv) {
		t.Helper()
		if !eqKV(got, want) {
			t.Fatalf("%s(%d,%d): got %v want %v", name, a, b, got, want)
		}
	}
	trunc := func(s []kv) []kv {
		if len(s) > limit {
			return s[:limit]
		}
		return s
	}
	asc := []kv(m)
	desc := reversed(asc)
	check("Ascend", collectN(limit, tr.Ascend), trunc(asc))
	check("Descend", collectN(limit, tr.Descend), trunc(desc))
	check("AscendRange", collect(func(f ItemIteratorG[kv]) { tr.AscendRange(ka, kb, f) }),
		m.filter(func(k int) bool { return k >= a && k < b }))
	check("AscendRangeN", collectN(limit, func(f ItemIteratorG[kv]) { tr.AscendRange(ka, kb, f) }),
		trunc(m.filter(func(k int) bool { return k >= a && k < b })))
	check("AscendLessThan", collectN(limit, func(f ItemIteratorG[kv]) { tr.AscendLessThan(ka, f) }),
		trunc(m.filter(func(k int) bool { return k < a })))
	check("AscendGreaterOrEqual", collectN(limit, func(f ItemIteratorG[kv]) { tr.AscendGreaterOrEqual(ka, f) }),
		trunc(m.filter(func(k int) bool { return k >= a })))
	check("DescendRange", collect(func(f ItemIteratorG[kv]) { tr.DescendRange(ka, kb, f) }),
		reversed(m.filter(func(k int) bool { return k <= a && k > b })))
	check("DescendRangeN", collectN(limit, func(f ItemIteratorG[kv]) { tr.DescendRange(ka, kb, f) }),
		trunc(reversed(m.filter(func(k int) bool { return k <= a && k > b }))))
	check("DescendLessOrEqual", collectN(limit, func(f ItemIteratorG[kv]) { tr.DescendLessOrEqual(ka, f) }),
		trunc(reversed(m.filter(func(k int) bool { return k <= a }))))
	check("DescendGreaterThan", collectN(limit, func(f ItemIteratorG[kv]) { tr.DescendGreaterThan(ka, f) }),
		trunc(reversed(m.filter(func(k int) bool { return k > a }))))
}

// runOps interprets data as a sequence of operations applied both to a set of
// btrees (sharing structure through Clone) and to reference models.
//
// The first byte selects the tree degree (data[0]%8) and the size of bulk
// operations (data[0]/8%3), which insert or delete up to 255*bulkScale keys at
// once.  Large bulk operations build trees several levels deep even at degree
// 64, at a cost in speed.
func runOps(t *testing.T, data []byte) {
	if len(data) == 0 {
		return
	}
	// Small degrees give deep trees with constant splits and merges; large
	// degrees give wide nodes where the binary searches matter.
	degrees := [...]int{2, 3, 4, 5, 8, 16, 32, 64}
	degree := degrees[int(data[0])%len(degrees)]
	bulkScales := [...]int{0, 1, 8}
	bulkScale := bulkScales[int(data[0])/len(degrees)%len(bulkScales)]
	data = data[1:]
	trees := []*BTreeG[kv]{NewWithFreeListG(degree, kvLess, NewFreeListG[kv](8))}
	models := []model{nil}
	serial := 0
	next := func() int {
		if len(data) == 0 {
			return 0
		}
		b := data[0]
		data = data[1:]
		return int(b)
	}
	// nextKey draws keys from two pools: half the time a dense range [0, 128),
	// where inserts and deletes keep colliding, and otherwise a sparse range
	// [128, 32896), which lets trees grow.
	nextKey := func() int {
		b := next()
		if b < 128 {
			return b
		}
		return 128 + (b-128)<<8 | next()
	}
	for steps := 0; len(data) > 0; steps++ {
		op := next()
		ti := next() % len(trees)
		tr, m := trees[ti], &models[ti]
		switch op % 14 {
		case 0, 1, 2, 3:
			serial++
			x := kv{key: nextKey(), val: serial}
			got, gok := tr.ReplaceOrInsert(x)
			want, wok := m.replaceOrInsert(x)
			if got != want || gok != wok {
				t.Fatalf("ReplaceOrInsert(%v): got %v,%v want %v,%v", x, got, gok, want, wok)
			}
		case 4, 5:
			k := nextKey()
			got, gok := tr.Delete(kv{key: k})
			want, wok := m.delete(k)
			if got != want || gok != wok {
				t.Fatalf("Delete(%d): got %v,%v want %v,%v", k, got, gok, want, wok)
			}
		case 6:
			got, gok := tr.DeleteMin()
			var want kv
			var wok bool
			if len(*m) > 0 {
				want, wok = m.delete((*m)[0].key)
			}
			if got != want || gok != wok {
				t.Fatalf("DeleteMin: got %v,%v want %v,%v", got, gok, want, wok)
			}
		case 7:
			got, gok := tr.DeleteMax()
			var want kv
			var wok bool
			if len(*m) > 0 {
				want, wok = m.delete((*m)[len(*m)-1].key)
			}
			if got != want || gok != wok {
				t.Fatalf("DeleteMax: got %v,%v want %v,%v", got, gok, want, wok)
			}
		case 8:
			if len(trees) < 8 {
				trees = append(trees, tr.Clone())
				models = append(models, m.clone())
			}
		case 9:
			k := nextKey()
			got, gok := tr.Get(kv{key: k})
			var want kv
			i, wok := m.find(k)
			if wok {
				want = (*m)[i]
			}
			if got != want || gok != wok || tr.Has(kv{key: k}) != wok {
				t.Fatalf("Get(%d): got %v,%v want %v,%v", k, got, gok, want, wok)
			}
		case 10:
			checkQueries(t, tr, *m, nextKey(), nextKey(), next()%16)
		case 11:
			if next()%4 == 0 {
				tr.Clear(next()%2 == 0)
				*m = nil
			}
		case 12:
			// Bulk insert a strided run of keys.  Single inserts alone cannot
			// build a tree several levels deep at large degrees, which is where
			// splits, steals and merges move children between internal nodes.
			base, n, stride := nextKey(), next()*bulkScale, 1+next()%4
			xs := make([]kv, n)
			for i := range xs {
				serial++
				xs[i] = kv{key: base + i*stride, val: serial}
			}
			want, wok := m.bulkInsert(xs)
			for i, x := range xs {
				got, gok := tr.ReplaceOrInsert(x)
				if got != want[i] || gok != wok[i] {
					t.Fatalf("bulk ReplaceOrInsert(%v): got %v,%v want %v,%v", x, got, gok, want[i], wok[i])
				}
			}
		case 13:
			// Bulk delete a run of keys, in a fuzzer-chosen direction, to
			// shrink a large tree by several levels.
			base, n, reverse := nextKey(), next()*bulkScale, next()%2 == 0
			lo, hi := base, base+n-1
			if reverse {
				lo, hi = base-n+1, base
			}
			for i := 0; i < n; i++ {
				k := lo + i
				if reverse {
					k = hi - i
				}
				var want kv
				j, wok := m.find(k)
				if wok {
					want = (*m)[j]
				}
				got, gok := tr.Delete(kv{key: k})
				if got != want || gok != wok {
					t.Fatalf("bulk Delete(%d): got %v,%v want %v,%v", k, got, gok, want, wok)
				}
			}
			m.deleteRange(lo, hi)
		}
		// Check structure periodically, and after every bulk operation, so a
		// broken invariant is reported close to the operation that broke it.
		if steps%32 == 0 || op%14 >= 12 {
			checkInvariants(t, tr)
		}
		if tr.Len() != len(*m) {
			t.Fatalf("Len: got %d want %d", tr.Len(), len(*m))
		}
	}
	checkFreelistDisjoint(t, trees)
	for i, tr := range trees {
		checkInvariants(t, tr)
		if got := collect(tr.Ascend); !eqKV(got, models[i]) {
			t.Fatalf("tree %d contents: got %v want %v", i, got, models[i])
		}
		gmin, gok := tr.Min()
		gmax, xok := tr.Max()
		if len(models[i]) == 0 {
			if gok || xok {
				t.Fatalf("Min/Max on empty tree returned ok")
			}
		} else if gmin != models[i][0] || gmax != models[i][len(models[i])-1] {
			t.Fatalf("Min/Max: got %v,%v", gmin, gmax)
		}
	}
}

func FuzzBTreeG(f *testing.F) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 32; i++ {
		b := make([]byte, 64+r.Intn(4096))
		r.Read(b)
		// Seeds use no or small bulk operations, to stay fast when run as
		// part of plain `go test`; the fuzzer can mutate them into large ones.
		b[0] %= 16
		f.Add(b)
	}
	f.Fuzz(runOps)
}

// TestRandomOps runs the fuzz body on random inputs so that it is exercised
// by plain `go test`.
func TestRandomOps(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	n := 500
	if testing.Short() {
		n = 50
	}
	for i := 0; i < n; i++ {
		b := make([]byte, 64+r.Intn(8192))
		r.Read(b)
		b[0] %= 8 // no bulk operations; TestRandomOpsLarge covers those
		runOps(t, b)
	}
}

// TestRandomOpsLarge runs long inputs with large bulk operations at each
// degree.  These build trees of thousands of items, deep enough (three or more
// levels even at degree 64) that splits, steals and merges move children
// between internal nodes, which small trees at large degrees never do.
func TestRandomOpsLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large random trees in short mode")
	}
	r := rand.New(rand.NewSource(7))
	for degreeIndex := 0; degreeIndex < 8; degreeIndex++ {
		b := []byte{2*8 + byte(degreeIndex)} // bulk scale index 2: large
		// Grow tree 0 to about 14000 items with eight bulk inserts of 2040
		// keys at stride 3, with bases 4096 apart, so that the random
		// operations that follow start from a deep tree and shrink it.
		for j := 0; j < 8; j++ {
			// op 12 (bulk insert), tree 0, sparse key 128+(16j)<<8, n 255*8, stride 1+2.
			b = append(b, 12, 0, 128+16*byte(j), 0, 255, 2)
		}
		tail := make([]byte, 4096)
		r.Read(tail)
		runOps(t, append(b, tail...))
	}
}

// checkFreelistDisjoint verifies that no node reachable from any of the trees
// (which all share one free list) is also sitting in that free list, i.e.
// that nothing has been freed while still in use, and that freed nodes have
// been cleared so they do not pin items for the garbage collector.
func checkFreelistDisjoint[T any](t *testing.T, trees []*BTreeG[T]) {
	t.Helper()
	live := map[*node[T]]bool{}
	var walk func(n *node[T])
	walk = func(n *node[T]) {
		live[n] = true
		for _, c := range n.children {
			walk(c)
		}
	}
	for _, tr := range trees {
		if tr.root != nil {
			walk(tr.root)
		}
	}
	fl := trees[0].cow.freelist
	for _, n := range fl.freelist {
		if live[n] {
			t.Fatalf("node in free list is still reachable from a tree")
		}
		if len(n.items) != 0 || len(n.children) != 0 || n.cow != nil {
			t.Fatalf("node in free list was not cleared: %d items, %d children, cow=%v", len(n.items), len(n.children), n.cow)
		}
	}
}
