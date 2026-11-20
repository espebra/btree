// Copyright 2014 Google Inc.
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
	"flag"
	"fmt"
	"hash/fnv"
	"math/rand"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"
)

// perm returns a random permutation of n Int items in the range [0, n).
func perm(rnd *rand.Rand, n int) (out []Item) {
	for _, v := range rnd.Perm(n) {
		out = append(out, Int(v))
	}
	return
}

var btreeDegree = flag.Int("degree", 32, "B-Tree degree")

var testSeed = flag.Int64("seed", 0, "seed for the tests' random inputs; 0 picks a new one each run")

// baseSeed is the seed for this run of the tests: -seed if given, otherwise
// one chosen from the clock.
var baseSeed = sync.OnceValue(func() int64 {
	if *testSeed != 0 {
		return *testSeed
	}
	return time.Now().UnixNano()
})

// newRand returns the source of random inputs for a test or benchmark.  Each
// test gets its own stream, derived from the run's seed and the test's name,
// so that what a test sees does not depend on which other tests ran or in what
// order (-shuffle, -run).  If the test fails, the seed is logged, and running
// it again with -seed reproduces its inputs exactly.
func newRand(tb testing.TB) *rand.Rand {
	tb.Helper()
	seed := baseSeed()
	tb.Cleanup(func() {
		if tb.Failed() {
			tb.Logf("random inputs were generated with -seed=%d", seed)
		}
	})
	h := fnv.New64a()
	h.Write([]byte(tb.Name()))
	return rand.New(rand.NewSource(seed ^ int64(h.Sum64())))
}

func ExampleBTree() {
	tr := New(*btreeDegree)
	for i := Int(0); i < 10; i++ {
		tr.ReplaceOrInsert(i)
	}
	fmt.Println("len:       ", tr.Len())
	fmt.Println("get3:      ", tr.Get(Int(3)))
	fmt.Println("get100:    ", tr.Get(Int(100)))
	fmt.Println("del4:      ", tr.Delete(Int(4)))
	fmt.Println("del100:    ", tr.Delete(Int(100)))
	fmt.Println("replace5:  ", tr.ReplaceOrInsert(Int(5)))
	fmt.Println("replace100:", tr.ReplaceOrInsert(Int(100)))
	fmt.Println("min:       ", tr.Min())
	fmt.Println("delmin:    ", tr.DeleteMin())
	fmt.Println("max:       ", tr.Max())
	fmt.Println("delmax:    ", tr.DeleteMax())
	fmt.Println("len:       ", tr.Len())
	// Output:
	// len:        10
	// get3:       3
	// get100:     <nil>
	// del4:       4
	// del100:     <nil>
	// replace5:   5
	// replace100: <nil>
	// min:        0
	// delmin:     0
	// max:        100
	// delmax:     100
	// len:        8
}

// TestItemForwarding checks that each method of the Item-based BTree gives
// the same result as the generic BTreeG it wraps.  The generic tree itself is
// tested by the rest of the suite; this only has to show that the wrappers
// forward arguments and results faithfully, including iterator early stops
// and nil for the generic API's (zero, false).
func TestItemForwarding(t *testing.T) {
	rnd := newRand(t)
	const n = 1000
	tr := New(3)
	g := NewOrderedG[int](3)
	toItems := func(vs []int) (out []Item) {
		for _, v := range vs {
			out = append(out, Int(v))
		}
		return
	}
	asItem := func(v int, ok bool) Item {
		if !ok {
			return nil
		}
		return Int(v)
	}
	same := func(what string, got, want interface{}) {
		t.Helper()
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: BTree gave %v, BTreeG gave %v", what, got, want)
		}
	}
	for _, v := range rnd.Perm(n) {
		v *= 2 // leave gaps, so that odd keys are absent
		same("ReplaceOrInsert", tr.ReplaceOrInsert(Int(v)), asItem(g.ReplaceOrInsert(v)))
	}
	same("ReplaceOrInsert existing", tr.ReplaceOrInsert(Int(10)), asItem(g.ReplaceOrInsert(10)))

	// Every iteration method, over random bounds (present and absent keys),
	// each with and without an early stop.
	for i := 0; i < 200; i++ {
		a, b, stop := rnd.Intn(2*n+4)-2, rnd.Intn(2*n+4)-2, rnd.Intn(2*n+4)-2
		var gotStop, wantStop int
		collectItems := func(f func(ItemIterator)) (out []Item) {
			f(func(x Item) bool {
				if int(x.(Int)) == stop {
					gotStop++
					return false
				}
				out = append(out, x)
				return true
			})
			return
		}
		collectInts := func(f func(ItemIteratorG[int])) (out []Item) {
			f(func(x int) bool {
				if x == stop {
					wantStop++
					return false
				}
				out = append(out, Int(x))
				return true
			})
			return
		}
		cases := []struct {
			name string
			item func(ItemIterator)
			gen  func(ItemIteratorG[int])
		}{
			{"Ascend", tr.Ascend, g.Ascend},
			{"Descend", tr.Descend, g.Descend},
			{"AscendRange", func(f ItemIterator) { tr.AscendRange(Int(a), Int(b), f) }, func(f ItemIteratorG[int]) { g.AscendRange(a, b, f) }},
			{"DescendRange", func(f ItemIterator) { tr.DescendRange(Int(a), Int(b), f) }, func(f ItemIteratorG[int]) { g.DescendRange(a, b, f) }},
			{"AscendLessThan", func(f ItemIterator) { tr.AscendLessThan(Int(a), f) }, func(f ItemIteratorG[int]) { g.AscendLessThan(a, f) }},
			{"AscendGreaterOrEqual", func(f ItemIterator) { tr.AscendGreaterOrEqual(Int(a), f) }, func(f ItemIteratorG[int]) { g.AscendGreaterOrEqual(a, f) }},
			{"DescendLessOrEqual", func(f ItemIterator) { tr.DescendLessOrEqual(Int(a), f) }, func(f ItemIteratorG[int]) { g.DescendLessOrEqual(a, f) }},
			{"DescendGreaterThan", func(f ItemIterator) { tr.DescendGreaterThan(Int(a), f) }, func(f ItemIteratorG[int]) { g.DescendGreaterThan(a, f) }},
		}
		for _, c := range cases {
			gotStop, wantStop = 0, 0
			same(fmt.Sprintf("%s(%d, %d) stopping at %d", c.name, a, b, stop), collectItems(c.item), collectInts(c.gen))
			same(c.name+" early stop", gotStop, wantStop)
		}
	}

	// Lookups and deletions, of present and absent keys.
	for i := 0; i < 2*n; i++ {
		k := rnd.Intn(2*n+4) - 2
		same("Get", tr.Get(Int(k)), asItem(g.Get(k)))
		same("Has", tr.Has(Int(k)), g.Has(k))
		same("Min", tr.Min(), asItem(g.Min()))
		same("Max", tr.Max(), asItem(g.Max()))
		switch rnd.Intn(4) {
		case 0:
			same("DeleteMin", tr.DeleteMin(), asItem(g.DeleteMin()))
		case 1:
			same("DeleteMax", tr.DeleteMax(), asItem(g.DeleteMax()))
		default:
			same("Delete", tr.Delete(Int(k)), asItem(g.Delete(k)))
		}
		same("Len", tr.Len(), g.Len())
	}

	// Clone gives an independent tree.
	c, cg := tr.Clone(), g.Clone()
	gmin, _ := g.Min()
	same("Clone Delete", c.Delete(tr.Min()), asItem(cg.Delete(gmin)))
	var all, allG []int
	tr.Ascend(func(x Item) bool { all = append(all, int(x.(Int))); return true })
	g.Ascend(func(x int) bool { allG = append(allG, x); return true })
	same("original after Clone Delete", toItems(all), toItems(allG))
	same("clone Len", c.Len(), cg.Len())

	// Clear empties the tree in both modes.
	tr.Clear(true)
	c.Clear(false)
	same("Len after Clear", []int{tr.Len(), c.Len()}, []int{0, 0})
	same("Min after Clear", tr.Min(), nil)
}

const benchmarkTreeSize = 10000

func BenchmarkInsert(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		tr := New(*btreeDegree)
		for _, item := range insertP {
			tr.ReplaceOrInsert(item)
			i++
			if i >= b.N {
				return
			}
		}
	}
}

func BenchmarkSeek(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	size := 100000
	insertP := perm(rnd, size)
	tr := New(*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tr.AscendGreaterOrEqual(Int(i%size), func(i Item) bool { return false })
	}
}

func BenchmarkDeleteInsert(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDeleteInsertCloneOnce(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	tr = tr.Clone()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDeleteInsertCloneEachTime(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr = tr.Clone()
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDelete(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	removeP := perm(rnd, benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New(*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Delete(item)
			i++
			if i >= b.N {
				return
			}
		}
		if tr.Len() > 0 {
			panic(tr.Len())
		}
	}
}

func BenchmarkGet(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	removeP := perm(rnd, benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New(*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Get(item)
			i++
			if i >= b.N {
				return
			}
		}
	}
}

func BenchmarkGetCloneEachTime(b *testing.B) {
	rnd := newRand(b)
	b.StopTimer()
	insertP := perm(rnd, benchmarkTreeSize)
	removeP := perm(rnd, benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New(*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr = tr.Clone()
			tr.Get(item)
			i++
			if i >= b.N {
				return
			}
		}
	}
}

type byInts []Item

func (a byInts) Len() int {
	return len(a)
}

func (a byInts) Less(i, j int) bool {
	return a[i].(Int) < a[j].(Int)
}

func (a byInts) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func BenchmarkAscend(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 0
		tr.Ascend(func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j++
			return true
		})
	}
}

func BenchmarkDescend(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 1
		tr.Descend(func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j--
			return true
		})
	}
}
func BenchmarkAscendRange(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 100
		tr.AscendRange(Int(100), arr[len(arr)-100], func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j++
			return true
		})
		if j != len(arr)-100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, j)
		}
	}
}

func BenchmarkDescendRange(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 100
		tr.DescendRange(arr[len(arr)-100], Int(100), func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j--
			return true
		})
		if j != 100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, j)
		}
	}
}
func BenchmarkAscendGreaterOrEqual(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 100
		k := 0
		tr.AscendGreaterOrEqual(Int(100), func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j++
			k++
			return true
		})
		if j != len(arr) {
			b.Fatalf("expected: %v, got %v", len(arr), j)
		}
		if k != len(arr)-100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, k)
		}
	}
}
func BenchmarkDescendLessOrEqual(b *testing.B) {
	rnd := newRand(b)
	arr := perm(rnd, benchmarkTreeSize)
	tr := New(*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Sort(byInts(arr))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 100
		k := len(arr)
		tr.DescendLessOrEqual(arr[len(arr)-100], func(item Item) bool {
			if item.(Int) != arr[j].(Int) {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j].(Int), item.(Int))
			}
			j--
			k--
			return true
		})
		if j != -1 {
			b.Fatalf("expected: %v, got %v", -1, j)
		}
		if k != 99 {
			b.Fatalf("expected: %v, got %v", 99, k)
		}
	}
}

const cloneTestSize = 10000

func BenchmarkDeleteAndRestore(b *testing.B) {
	rnd := newRand(b)
	items := perm(rnd, 16392)
	b.ResetTimer()
	b.Run(`CopyBigFreeList`, func(b *testing.B) {
		fl := NewFreeList(16392)
		tr := NewWithFreeList(*btreeDegree, fl)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dels := make([]Item, 0, tr.Len())
			tr.Ascend(ItemIterator(func(b Item) bool {
				dels = append(dels, b)
				return true
			}))
			for _, del := range dels {
				tr.Delete(del)
			}
			// tr is now empty, we make a new empty copy of it.
			tr = NewWithFreeList(*btreeDegree, fl)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`Copy`, func(b *testing.B) {
		tr := New(*btreeDegree)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dels := make([]Item, 0, tr.Len())
			tr.Ascend(ItemIterator(func(b Item) bool {
				dels = append(dels, b)
				return true
			}))
			for _, del := range dels {
				tr.Delete(del)
			}
			// tr is now empty, we make a new empty copy of it.
			tr = New(*btreeDegree)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`ClearBigFreelist`, func(b *testing.B) {
		fl := NewFreeList(16392)
		tr := NewWithFreeList(*btreeDegree, fl)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tr.Clear(true)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`Clear`, func(b *testing.B) {
		tr := New(*btreeDegree)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tr.Clear(true)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
}
