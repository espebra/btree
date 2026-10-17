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
func runOps(t *testing.T, data []byte) {
	if len(data) == 0 {
		return
	}
	// Small degrees give deep trees with constant splits and merges; large
	// degrees give wide nodes where the binary searches matter.
	degrees := [...]int{2, 3, 4, 5, 8, 16, 32, 64}
	degree := degrees[int(data[0])%len(degrees)]
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
	for len(data) > 0 {
		op := next()
		ti := next() % len(trees)
		tr, m := trees[ti], &models[ti]
		switch op % 12 {
		case 0, 1, 2, 3:
			serial++
			x := kv{key: next(), val: serial}
			got, gok := tr.ReplaceOrInsert(x)
			want, wok := m.replaceOrInsert(x)
			if got != want || gok != wok {
				t.Fatalf("ReplaceOrInsert(%v): got %v,%v want %v,%v", x, got, gok, want, wok)
			}
		case 4, 5:
			k := next()
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
			k := next()
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
			checkQueries(t, tr, *m, next(), next(), next()%16)
		case 11:
			if next()%4 == 0 {
				tr.Clear(next()%2 == 0)
				*m = nil
			}
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
		runOps(t, b)
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
