library(dplyr)
library(ggplot2)
library(caret)
library(MASS)

df <- read.csv('https://raw.githubusercontent.com/maile32/Customer-churn-prediction/main/Churn_Modelling.csv')

#Checks for null values
sapply(df, function(x) sum(is.na(x)))

#Exploratory Data Analysis
df$Exited <- as.factor(df$Exited)
plot1 <- ggplot(df, aes(x = Geography, fill = Exited)) + 
  geom_bar(aes(y = 100*(after_stat(count))/sum(after_stat(count))), position = 'dodge2') +
  ylab("Percentage")
plot1

plot2 <- ggplot(df, aes(x = Gender, fill = Exited)) + 
  geom_bar(aes(y = 100*(after_stat(count))/sum(after_stat(count))), position = 'dodge2') +
  ylab("Percentage")
plot2

plot3 <- ggplot(df, aes(x = HasCrCard, fill = Exited)) + 
  geom_bar(aes(y = 100*(after_stat(count))/sum(after_stat(count))), position = 'dodge2') +
  ylab("Percentage")
plot3

plot4 <- ggplot(df, aes(x = IsActiveMember, fill = Exited)) + 
  geom_bar(aes(y = 100*(after_stat(count))/sum(after_stat(count))), position = 'dodge2') +
  ylab("Percentage")
plot4

plot5 <- ggplot(df, aes(x = Exited, y = CreditScore)) + 
  geom_boxplot()
plot5

plot6 <- ggplot(df, aes(x = Exited, y = Age)) + 
  geom_boxplot()
plot6

plot7 <- ggplot(df, aes(x = Exited, y = Tenure)) + 
  geom_boxplot()
plot7

plot8 <- ggplot(df, aes(x = Exited, y = Balance)) + 
  geom_boxplot()
plot8

plot9 <- ggplot(df, aes(x = Exited, y = NumOfProducts)) + 
  geom_boxplot()
plot9

plot10 <- ggplot(df, aes(x = Exited, y = EstimatedSalary)) + 
  geom_boxplot()
plot10

#One-hot codes the categorical variable
df <- subset(df, select = -c(Surname,CustomerId,RowNumber))
dummy <- dummyVars(" ~ .", data=df)
final_df <- data.frame(predict(dummy, newdata = df))
final_df <- subset(final_df, select = -Exited.0)
colnames(final_df)[colnames(final_df) == 'Exited.1'] <- 'Exited'

#Check for correlation between numerical variables
corr.matrix <- cor(final_df)

#Splits the data into training and testing sets
train_row_num <- createDataPartition(final_df$Exited, p = 0.7, list = FALSE)
set.seed(2023)
train_set <- final_df[train_row_num,]
test_set <- final_df[-train_row_num,]

#Scales numerical variables
preprocess_params <- preProcess(train_set, method = 'range', list = FALSE)
train_set <- predict(preprocess_params, train_set)
test_set <- predict(preprocess_params, test_set)


#Logistic Regression
log_model <- glm(Exited ~., family = 'binomial', data = train_set)
print(summary(log_model))

#Assesses the predictive ability of the model
fitted.results <- predict(log_model,newdata=test_set,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misclass_error <- mean(fitted.results != test_set$Exited)
print(paste('Logistic Regression Accuracy',1-misclass_error))

#Creates a confusion matrix
print("Confusion Matrix for Logistic Regression")
table(test_set$Exited, fitted.results > 0.5)