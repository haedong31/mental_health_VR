library(tidyverse)


## custom functions -----
read_multi_csv <- function(path) {
  p <- dir(path, full.names = TRUE)
  ctl <- vector(mode = 'list', length = length(p))
  
  for (i in seq_along(p)) {
    tryCatch({
      f <- read_csv(p[i])
      ctl[[i]] <- f
    }, 
    error = function(e){ctl[[i]] <- NULL})
  }
  
  return(ctl)
}

text_prep <- function(x) {
  s <- vector(mode = 'list', length = length(x))
  
  for (i in seq_along(x)) {
    df_ori <- x[[i]]
    df <- filter(df_ori, speaker == 'PAR')
    u <- df$utterance
    
    u <- u %>% 
      str_trim(side = 'both') %>% # trim whitespace from start and end
      str_replace_all('[.?!]$', '') %>% # remove punctuation at end of sentence
      str_replace_all('[[:space:]]+', ' ') %>%  # normalize whitespace
      str_to_lower() %>% # lowercase
      str_trim(side = 'both') # trim whitespace from start and end
    
    # remove parentheses enclosing characters
    pidx <- str_locate(u, '\\([[:alpha:]^\\)]+\\)')
    for (j in seq_along(u)) {
      if (any(is.na(pidx[j, ]))) {
        next
      }
      
      a <- pidx[j, 1]
      b <- pidx[j, 2]
      str_sub(u[j], a, a) <- ''
      str_sub(u[j], b-1, b-1) <- ''
    }
    
    s[[i]] <- df$utterance
  }
  
  return(s)
}

## data pre-processing
# read data
cookie_control <- read_multi_csv('./data/Control/cookie')
cookie_dementia <- read_multi_csv('./data/Dementia/cookie')

cookie_control <- text_prep(cookie_control)
cookie_dementia <- text_prep(cookie_dementia)
