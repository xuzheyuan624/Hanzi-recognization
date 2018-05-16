Hanzi Recognization
===================
Requirement
-----------
Python = 3.6<br>
TensorFlow = 1.0.0<br>

Installation
-------------
Clone the repository<br>
```Shell
git clone https://github.com/xuzheyuan624/Hanzi-recognization.git
```

Datastes
--------
We use CASIA-HWDB (A Handwritten Chinese datasets made by CAS)<br>
You can get this dataset by:<br>
```Shell
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
```
unzip the dataset to $./data

Make tfrecord
-------------
This dataset has more than 3000 Chinese characters.<br>
If you just want to recognize a small part,you can modify the classes($./data/cfgs.py)<br>
Configure parameters in $./data/cfgs.py and modify the path and classes.<br>
```Shell
cd $./data
bash convert_train.sh
bash convert_test.sh
```
You can write pictures of the dataset like:
```Shell
cd $./data
python write_image.py
```
And you can find pictures in $./data/png

Train 
-----
```Shell
python train.py
```
And if you want to go on the last training:
```Shell
python train.py --restore_dir=weights(the weights path that you want to restore)
```
Test
----
``` Shell
python test.py --weights=ournet_final.ckpt --image=test.jpg
```

Eval
----
Choosing the wights that you want to evaluate
```Shell
python eval.py --weights=eval weights
```
You can find the resu;t in $./results
