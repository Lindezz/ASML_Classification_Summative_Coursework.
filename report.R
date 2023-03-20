#Read data
personal_loan = read.csv( file = "bank_personal_loan.csv", header=TRUE)
#Print basic information about the data
str(personal_loan)
summary(personal_loan$Personal.Loan)
# Check for missing values in the dataset
cat("Number of missing values",sum(is.na(personal_loan)))

###
# import all the library we need 
library(dplyr)
library(ggplot2)
library(tidyr)
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)
library(pROC)
library(e1071)
library(caret)
library(randomForest)
library(tidyverse)
library(tidytext)
###

###
#Data pre-processing
#Define a normalized function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
#Normalisation and factoring of data
data = personal_loan
data = data %>%
  mutate(Age  = normalize(Age),
         Experience = normalize(Experience),
         Income = normalize(Income),
         CCAvg = normalize(CCAvg),
         Mortgage = normalize(Mortgage),
         Education = as.factor(Education),
         Personal.Loan = as.factor(Personal.Loan),
         ZIP.Code  =  as.factor(cut(ZIP.Code, breaks = seq(from =90000, to = 97000, by = 1000),labels = FALSE, include.lowest = TRUE)),
         Securities.Account = as.factor(Securities.Account),
         CD.Account = as.factor(CD.Account),
         Online = as.factor(Online),
         CreditCard  = as.factor(CreditCard)
  )%>%
  as.data.frame()%>%
  na.omit(data())
###

###
#Data visualisation
#Use a box plot to get a first look at the relationship between all numerical variables and Personal.Loan
data %>%
  select(Personal.Loan,where(is.numeric)) %>%
  pivot_longer(cols = -Personal.Loan,
               names_to = "variable",
               values_to = "value")%>%
  ggplot(aes(x=Personal.Loan,y=value,fill= Personal.Loan))+
  geom_boxplot()+
  facet_wrap(~variable,scales = "free")+
  theme_test()


#A preliminary look with a bar chart at the relationship between all categorical variables and Personal.Loan
data %>%
  select(Personal.Loan,where(is.factor)) %>%
  pivot_longer(cols = -Personal.Loan,
               names_to = "variable",
               values_to = "value")%>%
  ggplot(aes(x=value,fill= Personal.Loan))+
  geom_bar(stat = "count", position = "fill")+
  facet_wrap(~variable,scales = "free")+
  labs( x = "value",y= "Percentage")
theme_test() 
###

###
## Splitting the data set
set.seed(123)
sample <-  sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.6,0.4))
train_data  <- data[sample, ]
test_data   <- data[!sample, ]
###

###
#Data modelling
#Logistic regression modelling of training set data
model_log <- glm(Personal.Loan~.,family = binomial(link = "logit"), data = train_data)
sort(model_log$coefficients,decreasing = TRUE)
#Predict and visualise the relationship between income, education and personal loans under a logistic regression model
pro_log = predict(model_log, newdata = test_data, type = "response")

test_data %>%
  mutate(Probability = pro_log)%>%
  ggplot(aes(x = Income,y=Probability,color= Education,size = Family ))+
  geom_point(alpha =0.5)+
  geom_hline(yintercept = 0.5,color = "blue")+
  labs(y =  "Probability of accpet the personal loan")+
  theme_test()

# Modeling the training set with SVM models in order to enrich the diversity of models
model_svm <- svm(Personal.Loan~., data = train_data)
model_svm

#Model the training set with a decision tree model and print the results
model_tree <- rpart(Personal.Loan ~ ., data = train_data , method = "class")
rpart.plot(model_tree)

#Model the training set with a random forest model and print the importance results between variables
model_forest <- randomForest(Personal.Loan ~ .,data = train_data, importance = TRUE)
varImpPlot(model_forest, n.var = min(nrow(model_forest$importance)), main = 'variable importance')
###

###
#Define a function that outputs the evaluation scores (Accuracy, precision, recall, f1_score, specificity) for each model
class_metric = function(true,predict){
  confusion_matrix <- confusionMatrix(predict,true)
  accuracy <- confusion_matrix$overall["Accuracy"]
  precision <- confusion_matrix$byClass["Pos Pred Value"]
  recall <- confusion_matrix$byClass["Sensitivity"]
  f1_score <- 2 * precision  * recall  / (precision  + recall )
  specificity = confusion_matrix$byClass["Specificity"]
  
  return(list(Accuracy = accuracy,
              Precision = precision,
              Recall =  recall,
              F1_score= f1_score,
              Specificity  = specificity 
              
  ))
}

#Use each model to predict the test set data separately
predictions_log = as.factor(ifelse(predict(model_log, newdata = test_data, type = "response") > 0.5, 1, 0))
predictions_svm <- predict(model_svm, test_data,type = "class")
predictions_tree <- predict(model_tree, test_data, type = "class")
predictions_rf <- predict(model_forest,test_data,type = "class")


#Use the defined function to obtain the evaluation scores for each model
log_metric = class_metric(predictions_log,test_data$Personal.Loan)
svm_metric = class_metric(predictions_svm,test_data$Personal.Loan)
tree_metric = class_metric(predictions_tree,test_data$Personal.Loan)
rf_metric = class_metric(predictions_rf,test_data$Personal.Loan)


#Visualisation of the evaluation scores for each model
bind_rows(unlist(log_metric),
          unlist(svm_metric),
          unlist(tree_metric),
          unlist(rf_metric)
)%>%
  rename(Accuracy = Accuracy.Accuracy ,Precision = "Precision.Pos Pred Value",Recall = Recall.Sensitivity,F1_score = "F1_score.Pos Pred Value", Specificity=Specificity.Specificity)%>%
  
  mutate(model = c("Logistic Regression",
                   "SVM",
                   "Decision Tree",
                   " Random Forest"
  ))%>%
  pivot_longer(cols = -model,
               names_to = "metric",
               values_to = "value")%>%
  mutate( model = reorder_within(x = model, by = value,within = metric))%>%
  ggplot(aes(x = model,y = value,fill = metric))+
  geom_col()+
  scale_x_reordered()+
  facet_wrap(~metric,scales = 'free')+
  labs( x = "Model",
        y = "Value",
        fill = "Model")+
  coord_flip()+
  theme_test()+
  theme(legend.position = "none")
###

###
#Perform PCA on the dataset and output the results ( retain 90% of the information )
data_slect <- subset(data, select = -c(ZIP.Code))
data_dummy <- model.matrix(~.-1, data = data_slect )
pca <- prcomp(data_dummy , scale. = TRUE)
variances <- pca$sdev ^ 2
total_variance <- sum(variances)
prop_variances <- variances / total_variance
cum_prop_variances <- cumsum(prop_variances)
n_components <- which.max(cum_prop_variances >= 0.9)
cat("In order to retain 90% of the information you need to retain the number of variables：",n_components )
###

###
#Grid search of random forests for hyperparameter tuning（Note: Grid search takes about 8-9 minutes.）

#Define the range of values for each hyperparameter
n_trees <- seq(from = 200, to = 800, by = 100)
max_depth <- seq(from = 5, to = 15, by =3 )
mtry <- seq(from = 4, to = 10, by = 1)
node_size <- seq(from = 5, to = 15, by = 5)
# Defining the cross-validation function
cv_func <- function(params) {
  rf_model <- randomForest(
    x=train_data[,!names(train_data) %in% "Personal.Loan"],
    y=train_data$Personal.Loan, 
    ntree = params[1], 
    maxdepth = params[2], 
    mtry = params[3], 
    nodesize = params[4]
  )
  train_pred <- predict(rf_model, newdata = train_data, type = "prob")[, 2]
  test_pred <- predict(rf_model, newdata = test_data, type = "prob")[, 2]
  train_auc <- roc(train_data$Personal.Loan, train_pred)$auc
  test_auc <- roc(test_data$Personal.Loan, test_pred)$auc
  return(list(train_auc=train_auc, test_auc=test_auc))
}

# Defining the parametric grid
params_grid <- expand.grid(n_trees, max_depth, mtry, node_size)

# Parameter tuning using cross-validation (Ignore the error alert)
cv_results <- apply(params_grid, 1, cv_func)



# Extraction of scores from the training and test sets
train_aucs <- sapply(cv_results, function(x) x$train_auc)
test_aucs <- sapply(cv_results, function(x) x$test_auc)

#Output optimal parameters and optimal learning rate
best_params <- params_grid[which.max(test_aucs), ]
best_auc <- max(test_aucs)
best_params
best_auc 

# Mapping the learning curve
learning_curve <- data.frame(
  n_trees,
  max_depth,
  mtry, 
  node_size,
  train_auc=train_aucs,
  test_auc=test_aucs
)



ggplot(learning_curve, aes(x=n_trees)) +
  geom_line(aes(y=train_auc, color="Train")) +
  geom_line(aes(y=test_auc, color="Test")) +
  scale_color_manual(values=c("Train"="blue", "Test"="red")) +
  labs(x="Number of Trees", y="AUC", color="") +
  theme_bw()

ggplot(learning_curve, aes(x=max_depth)) +
  geom_line(aes(y=train_auc, color="Train")) +
  geom_line(aes(y=test_auc, color="Test")) +
  scale_color_manual(values=c("Train"="blue", "Test"="red")) +
  labs(x="Max_depth", y="AUC", color="") +
  theme_bw()
ggplot(learning_curve, aes(x=mtry)) +
  geom_line(aes(y=train_auc, color="Train")) +
  geom_line(aes(y=test_auc, color="Test")) +
  scale_color_manual(values=c("Train"="blue", "Test"="red")) +
  labs(x="Number of mtry", y="AUC", color="") +
  theme_bw()
ggplot(learning_curve, aes(x=node_size)) +
  geom_line(aes(y=train_auc, color="Train")) +
  geom_line(aes(y=test_auc, color="Test")) +
  scale_color_manual(values=c("Train"="blue", "Test"="red")) +
  labs(x="Number of node_size", y="AUC", color="") +
  theme_bw()
###

###

#Modelling the adapted random forest model and comparing it with the previous initial model
set.seed(123)
model_forest_best<- randomForest(Personal.Loan~Income+CCAvg+Family+CD.Account+Education+Age+Experience+Mortgage+CreditCard,data = data, importance = TRUE,mtry =8 ,ntree = 500,max_depth=5,node_size=10)
model_forest <- randomForest(Personal.Loan ~ .,data = data, importance = TRUE)


model_forest_best
model_forest
plot(model_forest_best )
###
