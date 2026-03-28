studio:
	cargo run --bin imod-studio

test_viewer:
	cargo run --bin imod-viewer -- IMOD/Etomo/unitTestData/headerTest.st      

test_st:
	cargo run --bin mrcinfo -- IMOD/Etomo/unitTestData/headerTest.st

loc:
	find crates ui apps -name '*.rs' | xargs wc -l

gitaddall:
	git add crates apps
