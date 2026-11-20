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
	"fmt"
	"math"
	"strings"
	"testing"
)

// mustPanicWith calls f and fails unless it panics with a message containing
// want.
func mustPanicWith(t *testing.T, name, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Errorf("%s: expected panic", name)
			return
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, want) {
			t.Errorf("%s: panic %q does not mention %q", name, msg, want)
		}
	}()
	f()
}

// TestLessStrictWeakOrderingG checks that Less[T] is a strict weak ordering
// over float values including NaN, signed zeros and infinities: irreflexive,
// asymmetric, transitive, and with transitive incomparability.
func TestLessStrictWeakOrderingG(t *testing.T) {
	otherNaN := math.Float64frombits(math.Float64bits(math.NaN()) ^ 1)
	vals := []float64{math.NaN(), otherNaN, math.Inf(-1), -1, math.Copysign(0, -1), 0, 1, math.Inf(1)}
	less := Less[float64]()
	equiv := func(a, b float64) bool { return !less(a, b) && !less(b, a) }
	for _, a := range vals {
		if less(a, a) {
			t.Errorf("less(%v, %v) is true", a, a)
		}
		for _, b := range vals {
			if less(a, b) && less(b, a) {
				t.Errorf("less(%v, %v) and less(%v, %v) are both true", a, b, b, a)
			}
			for _, c := range vals {
				if less(a, b) && less(b, c) && !less(a, c) {
					t.Errorf("not transitive: %v < %v < %v", a, b, c)
				}
				if equiv(a, b) && equiv(b, c) && !equiv(a, c) {
					t.Errorf("incomparability not transitive: %v ~ %v ~ %v", a, b, c)
				}
			}
		}
	}
	if !less(math.NaN(), math.Inf(-1)) {
		t.Errorf("NaN should order before -Inf")
	}
	if !equiv(math.NaN(), otherNaN) {
		t.Errorf("NaNs should be equivalent")
	}
	if !equiv(0, math.Copysign(0, -1)) {
		t.Errorf("-0 and +0 should be equivalent")
	}
}

// TestNaNG is a regression test: with a plain '<' ordering, NaN appeared equal
// to every item, so inserting it replaced arbitrary items and lost them.
func TestNaNG(t *testing.T) {
	t.Run("float64", func(t *testing.T) { testNaN[float64](t) })
	t.Run("float32", func(t *testing.T) { testNaN[float32](t) })
}

func testNaN[F ~float32 | ~float64](t *testing.T) {
	nan := F(math.NaN())
	for _, degree := range []int{2, 3, 32} {
		tr := NewOrderedG[F](degree)
		for i := 0; i < 100; i++ {
			tr.ReplaceOrInsert(F(i))
			if i%10 == 0 {
				tr.ReplaceOrInsert(nan)
			}
		}
		checkInvariants(t, tr)
		if tr.Len() != 101 {
			t.Fatalf("degree %d: Len = %d, want 101 (100 numbers and one NaN)", degree, tr.Len())
		}
		if v, ok := tr.Min(); !ok || !math.IsNaN(float64(v)) {
			t.Fatalf("degree %d: Min = %v, want NaN", degree, v)
		}
		if !tr.Has(nan) || !tr.Has(50) {
			t.Fatalf("degree %d: Has(NaN) = %v, Has(50) = %v", degree, tr.Has(nan), tr.Has(50))
		}
		var got []F
		tr.AscendRange(nan, 3, func(v F) bool { got = append(got, v); return true })
		if len(got) != 4 || !math.IsNaN(float64(got[0])) || got[1] != 0 || got[3] != 2 {
			t.Fatalf("degree %d: AscendRange(NaN, 3) = %v", degree, got)
		}
		if v, ok := tr.Delete(nan); !ok || !math.IsNaN(float64(v)) {
			t.Fatalf("degree %d: Delete(NaN) = %v, %v", degree, v, ok)
		}
		if tr.Has(nan) || tr.Len() != 100 {
			t.Fatalf("degree %d: after Delete(NaN): Has %v, Len %d", degree, tr.Has(nan), tr.Len())
		}
		checkInvariants(t, tr)
	}
}

// TestSignedZeroG checks that -0 and +0 are one key.
func TestSignedZeroG(t *testing.T) {
	tr := NewOrderedG[float64](2)
	tr.ReplaceOrInsert(0)
	if old, ok := tr.ReplaceOrInsert(math.Copysign(0, -1)); !ok || old != 0 {
		t.Fatalf("inserting -0 over +0: got %v, %v", old, ok)
	}
	if tr.Len() != 1 {
		t.Fatalf("Len = %d, want 1", tr.Len())
	}
}

// TestNilItemPanics checks the documented panic on inserting a nil Item.  It
// used to be silently stored when the tree was empty, and only crash in a later
// comparison.
func TestNilItemPanics(t *testing.T) {
	tr := New(2)
	mustPanicWith(t, "empty tree", "nil Item", func() { tr.ReplaceOrInsert(nil) })
	if tr.Len() != 0 || tr.Min() != nil {
		t.Fatalf("nil insert into empty tree changed it: Len %d", tr.Len())
	}
	for i := 0; i < 10; i++ {
		tr.ReplaceOrInsert(Int(i))
	}
	mustPanicWith(t, "non-empty tree", "nil Item", func() { tr.ReplaceOrInsert(nil) })
	if tr.Len() != 10 {
		t.Fatalf("nil insert into non-empty tree changed it: Len %d", tr.Len())
	}
}

// TestZeroValueG checks that a zero-value tree reads as empty and that the
// operations that need a LessFunc panic with a helpful message rather than a
// nil pointer dereference.
func TestZeroValueG(t *testing.T) {
	var z BTreeG[int]
	if z.Len() != 0 || z.Has(1) {
		t.Fatal("zero tree should be empty")
	}
	if _, ok := z.Get(1); ok {
		t.Fatal("Get on zero tree")
	}
	if _, ok := z.Min(); ok {
		t.Fatal("Min on zero tree")
	}
	if _, ok := z.Max(); ok {
		t.Fatal("Max on zero tree")
	}
	if _, ok := z.Delete(1); ok {
		t.Fatal("Delete on zero tree")
	}
	if _, ok := z.DeleteMin(); ok {
		t.Fatal("DeleteMin on zero tree")
	}
	if _, ok := z.DeleteMax(); ok {
		t.Fatal("DeleteMax on zero tree")
	}
	called := false
	iter := func(int) bool { called = true; return true }
	z.Ascend(iter)
	z.AscendRange(0, 10, iter)
	z.AscendLessThan(10, iter)
	z.AscendGreaterOrEqual(0, iter)
	z.Descend(iter)
	z.DescendRange(10, 0, iter)
	z.DescendLessOrEqual(10, iter)
	z.DescendGreaterThan(0, iter)
	if called {
		t.Fatal("iterator called on zero tree")
	}
	z.Clear(true)
	z.Clear(false)

	mustPanicWith(t, "ReplaceOrInsert", "uninitialised", func() { z.ReplaceOrInsert(1) })
	mustPanicWith(t, "Clone", "uninitialised", func() { z.Clone() })

	var zi BTree
	if zi.Len() != 0 || zi.Get(Int(1)) != nil || zi.Min() != nil {
		t.Fatal("zero BTree should be empty")
	}
	mustPanicWith(t, "BTree.ReplaceOrInsert", "uninitialised", func() { zi.ReplaceOrInsert(Int(1)) })
}

// TestAllocsG pins the allocation counts of hot operations, which the
// implementation is tuned for and which benchmarks alone would not catch
// regressing.  If a change intentionally alters a count, update it here.
func TestAllocsG(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping allocation counts in short mode")
	}
	tr := NewWithFreeListG(32, Less[int](), NewFreeListG[int](1024))
	for i := 0; i < 10000; i++ {
		tr.ReplaceOrInsert(i)
	}
	small := NewWithFreeListG(4, Less[int](), NewFreeListG[int](1024))
	for i := 0; i < 500; i++ {
		small.ReplaceOrInsert(i)
	}
	iter := func(int) bool { return true }
	stopAt := func(n int) ItemIteratorG[int] {
		return func(v int) bool { return v != n }
	}
	cases := []struct {
		name string
		max  float64
		f    func()
	}{
		{"Get", 0, func() { tr.Get(5000) }},
		{"Has", 0, func() { tr.Has(-1) }},
		{"Min/Max", 0, func() { tr.Min(); tr.Max() }},
		{"Ascend", 0, func() { tr.Ascend(stopAt(500)) }},
		{"AscendRange", 0, func() { tr.AscendRange(100, 600, iter) }},
		{"AscendLessThan", 0, func() { tr.AscendLessThan(300, iter) }},
		{"AscendGreaterOrEqual", 0, func() { tr.AscendGreaterOrEqual(9700, iter) }},
		{"Descend", 0, func() { tr.Descend(stopAt(9500)) }},
		{"DescendRange", 0, func() { tr.DescendRange(600, 100, iter) }},
		{"DescendLessOrEqual", 0, func() { tr.DescendLessOrEqual(300, iter) }},
		{"DescendGreaterThan", 0, func() { tr.DescendGreaterThan(9700, iter) }},
		{"ReplaceOrInsert existing", 0, func() { tr.ReplaceOrInsert(5000) }},
		{"Delete+ReplaceOrInsert", 0, func() { tr.Delete(5000); tr.ReplaceOrInsert(5000) }},
		{"DeleteMin+ReplaceOrInsert", 0, func() { tr.DeleteMin(); tr.ReplaceOrInsert(0) }},
		{"DeleteMax+ReplaceOrInsert", 0, func() { tr.DeleteMax(); tr.ReplaceOrInsert(9999) }},
		// Clone allocates both copy-on-write contexts together.
		{"Clone", 1, func() { tr.Clone() }},
		// Refilling from the free list should reuse nodes, not allocate them.
		{"Clear+refill", 2, func() {
			small.Clear(true)
			for i := 0; i < 500; i++ {
				small.ReplaceOrInsert(i)
			}
		}},
	}
	for _, c := range cases {
		if got := testing.AllocsPerRun(50, c.f); got > c.max {
			t.Errorf("%s: %v allocs per run, want <= %v", c.name, got, c.max)
		}
	}
	checkInvariants(t, tr)
	checkInvariants(t, small)
}
