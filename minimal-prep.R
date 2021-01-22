library(tidyverse)


## custom functions -----
read_multi_csv <- function(path) {
  p <- dir(path, full.names = TRUE)
  f <- dir(path, full.names = FALSE)
  
  ff <- f %>%
    str_split('\\.') %>% # split by '.'
    map(function(x){x[1]}) %>% # select first element (drop extension '.csv')
    flatten_chr() # list to vector
  
  contl <- vector(mode = 'list', length = length(p))
  names(contl) <- ff
  
  for (i in seq_along(p)) {
    tryCatch({
      f <- read_csv(p[i])
      contl[[i]] <- f
    }, 
    error = function(e){contl[[i]] <- NULL})
  }
  
  return(contl)
}

text_prep <- function(x) {
  s <- vector(mode = 'list', length = length(x))
  
  for (i in seq_along(x)) {
    df_ori <- x[[i]]
    df <- filter(df_ori, speaker == 'PAR')
    u <- df$utterance
    
    u <- u %>% 
      str_trim(side = 'both') %>% # trim whitespace from start and end
      # str_replace_all('[.?!]$', '') %>% # remove punctuation at end of sentence
      str_replace_all('[[:space:]]+', ' ') %>%  # normalize whitespace
      # str_to_lower() %>% # lowercase
      str_trim(side = 'both') # trim whitespace from start and end
    
    # remove parentheses enclosing characters
    pidx <- str_locate(u, '\\([[:alpha:]^\\)]+\\)')
    for (j in seq_along(u)) {
      if (any(is.na(pidx[j, ]))) {next}
      
      a <- pidx[j, 1]
      b <- pidx[j, 2]
      str_sub(u[j], a, a) <- ''
      str_sub(u[j], b-1, b-1) <- ''
    }
    
    s[[i]] <- u
  }
  
  return(s)
}

write_sen_list <- function(x, type) {
  if (type == 'control') {
    base_dir <- './data/cookie_minimal_prep/control'
    dir.create(base_dir, showWarnings = FALSE, recursive = TRUE)
    f <- names(x)
  } else if (type == 'experimental') {
    base_dir <- './data/cookie_minimal_prep/dementia'
    dir.create(base_dir, showWarnings = FALSE, recursive = TRUE)
    f <- names(x)
  } else if (is.null(names(x))) {stop('Unnamed input; no file names')
  } else {stop('Unknown group')}
  
  for (i in seq_along(x)) {
    sentence <- x[[i]]
    tryCatch({
      fp <- file.path(base_dir, str_c(f[i], '.txt'))
      write_lines(sentence, fp) # write sentences
    },
    warning = function(w){
      cat(sprintf('\n Warning at %d \n', i))
      message(w)
      },
    error = function(e){
      cat(sprintf('\n Error at %d \n', i))
      message(e)
      })
  }
}


## main -----
# read data
cookie_control <- read_multi_csv('./data/Control/cookie')
cookie_dementia <- read_multi_csv('./data/Dementia/cookie')

prep_cookie_control <- text_prep(cookie_control)
prep_cookie_dementia <- text_prep(cookie_dementia)
names(prep_cookie_control) <- names(cookie_control)
names(prep_cookie_dementia) <- names(cookie_dementia)

write_sen_list(prep_cookie_control, type = 'control')
write_sen_list(prep_cookie_dementia, type = 'experimental')
