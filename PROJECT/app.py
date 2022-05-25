from re import M
from flask import Flask,render_template,request,redirect,url_for

import datetime
import tensorflow as tf 
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas  as pd 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb 
import cv2

app=Flask(__name__)

def create_dataframe(img,comp,cost,min_cover,max_cover,exp):
    dic={
        "Image_path":[img],
        "Insurance_company":[comp],
        "Cost_of_vehicle":[cost],
        "Min_coverage":[min_cover],
        "Max_coverage":[max_cover],
        "Expiry_date":[exp]
        }
    df=pd.DataFrame(dic) 
    print(df.head())
    return df 
def check_date(year,month,date):
    curr_date=datetime.datetime.now()
    given_date=datetime.datetime(year,month,date)
    return given_date>=curr_date

def add_condition(df,cond):
    df['Condition']=[cond]
    print(df.head())
    return df 
def preprocess_df(df):
    encode=LabelEncoder()
    encode.classes_=np.load("Model/classes.npy",allow_pickle=True)
    encode_dic={val:i for i,val in enumerate(list(encode.classes_))}
    df['Insurance_company']=df['Insurance_company'].map(encode_dic)
    ctype = df["Expiry_date"].str.split("-",expand = True)
    df["Year"]=ctype[0]
    df["Month"]=ctype[1]
    df["Day"]=ctype[2]
    df = df.astype({'Year':'int64','Month':'int64','Day':'int64'})
    df=df.drop("Expiry_date",1)
    df=df.drop("Image_path",1)
    print(df.head())
    return df 

def predict_amount(df):
    file=open("Model/model1","rb")
    pred=pickle.load(file)
    d_test = xgb.DMatrix(df)
    # pred = xgb.XGBClassifier();
    # pred.load_model("Model/xgb_model.json")
    p_test = pred.predict(d_test)
    return p_test[0]


    
def predict_condition(df,img_path):
    tf_model = tf.keras.models.load_model('E:\Pro\PROJECT\Model\cond_model\my-model.h5')
    image_path = img_path
    x_test=[]
    test_image=cv2.imread(image_path)
    print(test_image)
    img=cv2.resize(test_image,(244,244))
    x_test.append(img)
    x_test=np.array(x_test)
    x_test=x_test/255
    predicted=tf_model.predict(x_test)
    out = np.argmax(predicted)
    return out
#     return out 
# def predict_condition(df):
#     tf_model = tf.keras.models.load_model('Model/my_model.h5')
#     test_datagen = image.ImageDataGenerator(rescale = 1.0/255.)
#     test_generator = test_datagen.flow_from_dataframe( dataframe = df, target_size = (224,224),
#                                             x_col = 'Image_path', y_col = None,
#                                              batch_size = 64,class_mode = None)
#     preds = (tf_model.predict(test_generator)>0.5).astype("int32")
#     return preds[0][0]

def final(dict_file,img_path):
    # img="/home/rishi/ML/Project2/dataset/testImages/"+dict_file.get("image_file")
    comp=dict_file.get("comp")
    cost=int(dict_file.get("cost"))
    min_cover=int(dict_file.get("min_cover"))
    max_cover=int(dict_file.get("max_cover"))
    exp=dict_file.get("exp")
    df=create_dataframe(img_path,comp,cost,min_cover,max_cover,exp)
    # print(predict_condition(df))
    year,month,date=map(int,exp.split('-'))
    if not check_date(year,month,date):
        return [0,0]
    # cond=predict_condition(df)
    cond=predict_condition(df,img_path)
    if cond==1:
        df=add_condition(df,cond)
        df=preprocess_df(df)
        pred_amt=predict_amount(df)
    else:
        pred_amt=0
    print("\n\n\n**********Condition",cond)
    print("**********Amount",pred_amt)
    return [cond,pred_amt]

@app.route('/')
def index_fun():
    return render_template("index.html")

@app.route('/result',methods=['GET','POST'])
def result_fun():
    if request.method=='POST':
        result=request.form 
        f=request.files['image_file']
        img_path="Upload/"+f.filename
        f.save(img_path)
        # curr_url=request.url
        cond,pred_amt=final(result,img_path)
        return render_template("result.html",condition=cond,Amount=pred_amt)
    else:
        return redirect('/')

if __name__ == '__main__':
   app.run()