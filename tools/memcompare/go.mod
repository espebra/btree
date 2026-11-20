module github.com/espebra/btree/tools/memcompare

go 1.27

require (
	github.com/espebra/btree v0.0.0
	github.com/petar/GoLLRB v0.0.0-20210522233825-ae3b015fd3e9
)

// Always build against the btree in this repository.
replace github.com/espebra/btree => ../..
