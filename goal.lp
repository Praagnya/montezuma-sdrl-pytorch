#program check(k).
:- query(k), cost(C,k), C <= 0.
:- query(k), cost(0,k).