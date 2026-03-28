studio:
	cargo run --bin imod-studio

test_viewer:
	cargo run --release --bin imod-viewer -- testdata/tutorialData/BBa.st   # IMOD/Etomo/unitTestData/headerTest.st      

test_st:
	cargo run --bin mrcinfo -- IMOD/Etomo/unitTestData/headerTest.st

loc:
	find crates ui apps -name '*.rs' | xargs wc -l


loc_orig:
	find IMOD -name '*.c' | xargs wc -l
	find IMOD -name '*.cpp' | xargs wc -l
	find IMOD -name '*.h' | xargs wc -l

gitaddall:
	git add crates apps

get_test_data:
	wget https://bio3d.colorado.edu/imod/files/tutorialData-1K.tar.gz --no-check-certificate

