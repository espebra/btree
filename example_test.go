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

//go:build go1.18
// +build go1.18

package btree_test

import (
	"fmt"

	"github.com/google/btree"
)

func ExampleNewOrderedG() {
	tr := btree.NewOrderedG[int](32)
	for _, v := range []int{5, 1, 4, 2, 3} {
		tr.ReplaceOrInsert(v)
	}
	tr.AscendRange(2, 5, func(v int) bool {
		fmt.Println(v)
		return true
	})
	// Output:
	// 2
	// 3
	// 4
}

type account struct {
	id      string
	balance int
}

func ExampleNewG() {
	// Order accounts by id.  Lookups and deletions take an account with only
	// the id set.
	tr := btree.NewG(32, func(a, b account) bool { return a.id < b.id })
	tr.ReplaceOrInsert(account{"bob", 10})
	tr.ReplaceOrInsert(account{"alice", 20})

	if old, ok := tr.ReplaceOrInsert(account{"bob", 15}); ok {
		fmt.Println("replaced", old)
	}
	if a, ok := tr.Get(account{id: "alice"}); ok {
		fmt.Println("found", a)
	}
	tr.Descend(func(a account) bool {
		fmt.Println(a.id, a.balance)
		return true
	})
	// Output:
	// replaced {bob 10}
	// found {alice 20}
	// bob 15
	// alice 20
}

func ExampleBTreeG_Clone() {
	tr := btree.NewOrderedG[int](32)
	for i := 0; i < 5; i++ {
		tr.ReplaceOrInsert(i)
	}
	// Clone is O(1); the two trees share structure until one of them writes.
	snapshot := tr.Clone()
	tr.Delete(2)
	tr.ReplaceOrInsert(10)

	fmt.Println("tree:", tr.Len(), "snapshot:", snapshot.Len())
	fmt.Println(snapshot.Has(2), tr.Has(2))
	// Output:
	// tree: 5 snapshot: 5
	// true false
}
