CXX = g++
OPTFLAGS = -O3 -fopenmp
CXXFLAGS = -I/opt/OpenBLAS/include 
LIBS = -I/opt/OpenBLAS/include -I/usr/local/lib/R/include -L /opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -L/usr/local/lib -lRmath  -lcholmod -lsuitesparseconfig -lcolamd -lcamd -lcholmod -lamd -lccolamd -lmetis

gauss_nngp: gauss_nngp.o kvpar.o
	$(CXX) $(OPTFLAGS) -o gauss_nngp gauss_nngp.o kvpar.o $(LIBS) 

gauss_nngp.o: gauss_nngp.cpp
	$(CXX) $(OPTFLAGS) gauss_nngp.cpp -c $(LIBS)

kvpar.o: ../libs/kvpar.cpp
	$(CXX) $(OPTFLAGS) ../libs/kvpar.cpp -c

clean:
	rm -rf *o gauss_nngp
