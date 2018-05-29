cd ./MaltOptimizer-1.0.3 &&
source temp-python2/bin/activate &&
python -V &&
java -jar MaltOptimizer.jar -p 1 -m ../maltparser-1.9.1/maltparser-1.9.1.jar -c ../input/kaist_train.conll &&
java -jar MaltOptimizer.jar -p 2 -m ../maltparser-1.9.1/maltparser-1.9.1.jar -c ../input/kaist_train.conll && #[-v <validation method>] &&
java -jar MaltOptimizer.jar -p 3 -m ../maltparser-1.9.1/maltparser-1.9.1.jar -c ../input/kaist_train.conll && #[-v <validation method>] &&
deactivate
