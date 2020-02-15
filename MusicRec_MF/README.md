# https://github.com/wolfecameron/music_recommendation/blob/master/music_ratings_learner.ipynb
## 모델
두 개의 임베딩 매트릭스를 선언한다.   
nn.Embedding(총 유저 수, 임베딩 시킬 벡터의 차원) 총 유저에 대해 해당 차원 만큼 임베딩 매트릭스를 생성함.   
bias 를 추가한다.   
해당 인덱스에 위치하는 유저, 아티스트 두 벡터를 가져온다.   
내적해서 rating을 예측한다.   
Bias 추가한 것을 더한다.   

## 데이터 로더
Method for getting a data loader from a csv file   
Create custom class for pytorch data set   
Initialize DataFrame   
get total number of samples   
get data sample from dataset   

## 메인 함수
Read in data frame created by EDA notebook   
separate music data into separate training and testing files   
get seperate dataframes with train and test data   
write these train and test data to separate csv file   
find number of artists and users being used 1   
create data loaders for training and test data   
declare the size of the embeddings to be used   
Find the optimal learning rate with chich to begin training   
run model   
go to the next learning rate   
change the learning rate in the optimizer   
declare training constants   
declare loss criterion for the model : MSELoss
create model and optimzer : SGD
Create training loop to train thre recommender model on test data   
loop over the dataset multiple times
