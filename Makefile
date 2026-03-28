loc:
	find crates ui apps -name '*.rs' | xargs wc -l

gitaddall:
	git add crates apps
