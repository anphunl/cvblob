# g++ -fPIC -I/usr/include/python2.7 -I/usr/include -I/usr/local/include -c _cvblob.C
# g++ -shared _cvblob.o -L/usr/lib -L/usr/local/lib -lopencv_core -lcvblob -lboost_python -L/usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7 -o _cvblob.so


g++ -fPIC -I/usr/include/python2.7 -I/usr/include -Ipyboostcvconverter/include -I/usr/local/include -c _cvblob.cpp
g++ -shared _cvblob.o -L/usr/lib -L/usr/local/lib -Lpyboostcvconverter/ -lopencv_core -lcvblob -lpbcvt -lboost_python -L/usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7 -o _cvblob.so
