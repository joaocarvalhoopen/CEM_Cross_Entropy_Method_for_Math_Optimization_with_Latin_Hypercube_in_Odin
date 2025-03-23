all:
	odin build . -out:cem_in_odin.exe -o:speed

opti_max:	
	odin build . -out:cem_in_odin.exe -o:aggressive -microarch:native -no-bounds-check -disable-assert -no-type-assert

clean_odin:
	rm -f cem_in_odin.exe