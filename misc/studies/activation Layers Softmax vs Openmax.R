# --------------------------------------------------------- #
# -- Alg 2
# https://arxiv.org/pdf/1511.06233.pdf
# --------------------------------------------------------- #
rm(list = ls())
require(tidyverse)
require(purrr)
library(fitdistrplus)

source("/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my weibull EVM/quality_metrics.R")

da = read.csv("/Users/leonardomarques/Downloads/total_da_2.csv")
da$pred = da[c("pred_0", "pred_1")] %>%
  apply(1, function(xx) which.max(xx) - 1)

da %>% head()

da %>% ggplot(aes(
  x = x1
  , y = x2
  , col = as.character(y))
  ) +
  geom_point()


train_da = da%>% filter(split == 'train')
val_da = da%>% filter(split == 'val')
test_da = da%>% filter(split == 'test')

# ---------------------------------------------------------------- #
# -- model
# ---------------------------------------------------------------- #
# -------------------- #
# -- MAVs
# -------------------- #
tail_size = 10 # Number of observations to use when estimating the rejection boundaries
classes_to_revise = -1 # number of classes to use when predicting.

mavs = split(train_da, train_da$y) %>%
  lapply(function(aux_df){
    # aux_df = split(train_da, train_da$y)[[1]]
    # aux_df = split(train_da, train_da$y)[[2]]
    # -- get correct predictions
    idx_correct = aux_df$y == aux_df$pred
    correct_df = aux_df[idx_correct, ]
    
    
    mav = correct_df[c('activation_vector_0', 'activation_vector_1')] %>%
      apply(2, mean)
    mav = data.frame(mav) %>% t()
    
    # dists = proxy::dist(mav, correct_df[c('x1', 'x2')], method = "euclidean")
    dists = proxy::dist(mav
                        , correct_df[c('activation_vector_0', 'activation_vector_1')]
                        , method = "euclidean")
    dists = sort(dists)
    longest_dists = tail(dists, tail_size)
    
    mle_wieb_nocens_fit <- fitdistrplus::fitdist(data=longest_dists, distr = "weibull")
    shape_hat <- mle_wieb_nocens_fit$estimate["shape"]
    scale_hat <- mle_wieb_nocens_fit$estimate["scale"]
    weibull_pars = c(shape_hat, scale_hat)
    weibull_pars['tau'] = 0 # I am using the 2 parametric weibull
    
    # plot(ecdf(longest_dists))
    # curve(pweibull(
    #     x
    #     , shape = weibull_pars['shape']
    #     , scale = weibull_pars['scale']
    #     , lower.tail = TRUE
    #     )
    #     , add = TRUE
    # )
    
    result = list(
      mav = mav
      , weibull_pars = weibull_pars
      )
    
    # ---------------------------- #
    # -- returns
    # ---------------------------- #
    return(result)
    
  })

mavs$`0`$mav
str(mavs, 2)

# -------------------------------------------------------- #
# -------------------------------------------------------- #
# -- OpenMax probability estimation
# "revising" all classes. 
# -------------------------------------------------------- #
# -------------------------------------------------------- #

# -- calculate distances
prediction_da = bind_rows(train_da, val_da, test_da)

i_class = 0
while (i_class < length(mavs)) {
  i_class = i_class + 1
  
  # prediction_da %>% ggplot(aes(
  #   x = activation_vector_0
  #   , y = activation_vector_1
  #   , col = as.character(y))
  # ) +
  #   geom_point() +
  #   geom_point(
  #     data=data.frame(mavs[[i_class]]$mav)
  #     , aes(x = activation_vector_0
  #           , y = activation_vector_1)
  #     , inherit.aes = FALSE
  #     )
  
  # head(prediction_da[c('pred_0', 'pred_1')])
  
  weibull_pars = mavs[[i_class]]$weibull_pars
  
  dist_to_class = proxy::dist(
    mavs[[i_class]]$mav
    , prediction_da[c('activation_vector_0', 'activation_vector_1')]
    , method = "euclidean")
  dist_to_class = dist_to_class %>% t()
  
  prediction_da[paste0('dist_class_', i_class-1)] = dist_to_class[, 1]
  # head(prediction_da)
  
  
}


# -------------------------------- #
# -- weigths
# -------------------------------- #

i_class = 0
while (i_class < length(mavs)) {
  i_class = i_class + 1
  
  weights_ = 1:nrow(prediction_da) %>%
    sapply(function(ii){
      # ii = 1
      aux_df = prediction_da[ii, ]
      
      if(classes_to_revise == -1){
        ALPHA = length(mavs)
      } else {
        ALPHA = classes_to_revise
        }
      
      weibull_pars = mavs[[i_class]]$weibull_pars
      # mavs[[i_class]]$mav
      
      activation_order =  order(
        aux_df[c('activation_vector_0', 'activation_vector_1')]
        , decreasing = TRUE)[i_class]
      
      # -- distance to MAV
      distance_ = aux_df[, c('dist_class_0', 'dist_class_1')[i_class]]
      
      # -- weigths
      ws_ = 1 - ((ALPHA - activation_order)/ALPHA) * 
        pweibull(
          distance_
          , shape = weibull_pars['shape']
          , scale = weibull_pars['scale']
          , lower.tail = TRUE
        )
      # The Meta-Recognition probability (CDF of a Weibull) is a monotonically increasing function
      # Section 2.4 theorem 1
      
      return(ws_)
      
    })
  
  prediction_da[paste0('weigth_', i_class-1)] = weights_
  
}


# head(prediction_da)

# ---------------------------------------- #
# -- revise activation vector, line 5
# ---------------------------------------- #

aux_revised = prediction_da[c('activation_vector_0', 'activation_vector_1')]*
  prediction_da[c('weigth_0', 'weigth_1')]

names(aux_revised) = gsub(
  names(aux_revised)
  , patt = 'activation_vector_'
  , repl='revised_')

prediction_da = cbind(prediction_da, aux_revised)
head(prediction_da)


# -------------------------------- #
# -- Define V0, line 6
# -------------------------------- #

prediction_da[c('v_zero_0', 'v_zero_1')] = prediction_da[c('revised_0', 'revised_1')]*(1-prediction_da[c('weigth_0', 'weigth_1')])

# -------------------------------- #
# -- OpenMax
# -------------------------------- #

soft_max = prediction_da[c('v_zero_0', 'v_zero_1')] %>%
  apply(1, function(xx){
    exp(xx) / sum(exp(xx))
  }) %>% t()

prediction_da[c('open_max_0', 'open_max_1')] = soft_max

open_max_pred = prediction_da[c('open_max_0', 'open_max_1')] %>%
  apply(1, function(xx){
    which.max(xx) - 1
  })
prediction_da['open_max_pred'] = open_max_pred

pred_prob = prediction_da[c('open_max_0', 'open_max_1')] %>%
  apply(1, function(xx){
    max(xx)
  })
prediction_da['pred_prob'] = pred_prob


prediction_da %>%
  ggplot(aes(
    x = open_max_0
    , y = open_max_1
    , col = as.character(y)
  )) + 
  geom_point()


# ---------------------------------------- #
# - quality
# ---------------------------------------- #

qualities_list = c(0:100/100) %>%
  lapply(function(threshold){
    # threshold = 0.5
    # threshold = 0.71
    # threshold = 0.97
    # print(threshold)
    aux_df = prediction_da
    
    aux_df$predictions = aux_df$open_max_pred
    
    id_reject = aux_df$pred_prob < threshold
    aux_df$predictions[id_reject] = -1
    # table(aux_df$predictions)
    
    # -- precision
    precision = os_precision(
      aux_df$y
      , aux_df$predictions
      , unknown_label = -1
      , labels = sort(unique(prediction_da$y))
    )
    
    # -- recall
    recall = os_recall(
      aux_df$y
      , aux_df$predictions
      , unknown_label = -1
      , labels = sort(unique(prediction_da$y))
    )
    
    # -- accuracy 
    # debugonce(os_accuracy)
    accuracy = os_accuracy(
      aux_df$y
      , aux_df$predictions
      , unknown_label = -1
      , labels=sort(unique(prediction_da$y))
      , lambda = .5  # regularization parameter (Known accuracy weight)
    )
    
    # -- True negative rate
    true_neg_rate = os_true_negative_rate(
      aux_df$y
      , aux_df$predictions
      , unknown_label =-1
      , labels=sort(unique(prediction_da$y))
    )
    
    # -- rejection rate
    rejection_rate = sum(aux_df$predictions == -1) / nrow(aux_df)
    
    
    # -- results -- #
    results_df = data.frame(
      metric = c('precision'
                 , 'recall'
                 , 'accuracy'
                 , 'true_negative_rate'
                 , 'rejection_rate')
      , value = c(precision
                  , recall
                  , accuracy
                  , true_neg_rate
                  , rejection_rate)
    )
    
  })



quality_df = qualities_list %>%
  bind_rows(.id = 'threshold')

quality_df$threshold = as.integer(quality_df$threshold) - 1

quality_df %>%
  ggplot(aes(
    x = threshold
    , y = value
    , col = metric
    , size = metric 
  )) + 
  geom_line(alpha = 0.3) + 
  facet_wrap(~metric, ncol = 3) + 
  theme(legend.position = 'None')


