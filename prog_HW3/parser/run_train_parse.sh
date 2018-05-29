cd output
java -jar ../maltparser-1.9.1/maltparser-1.9.1.jar -c korea_model -i ../input/kaist_train_new.conll -m learn &&
java -jar ../maltparser-1.9.1/maltparser-1.9.1.jar -c korea_model -i ../input/kaist_test_new.conll -o test_out.conll -m parse &&
java -jar ../maltparser-1.9.1/maltparser-1.9.1.jar -c korea_model -i ../input/kaist_dev_new.conll -o dev_out.conll -m parse &&
cd ..
java -jar ./malteval/lib/MaltEval.jar -g ./input/kaist_dev_new.conll -s ./output/dev_out.conll
