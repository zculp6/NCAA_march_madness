library(dplyr)
library(httr)
library(glue)
library(readr)
library(lubridate)
library(cbbdata)

cbd_torvik_team_factors <- function(year = NULL, gender = "M") {
  suppressWarnings({
    # default PC user-agent gets blocked on barttorvik.com
    local_options(HTTPUserAgent='CBB-DATA')
    
    # Validate gender input
    if (!(gender %in% c("M", "W"))) {
      cli::cli_abort("Invalid gender input. Use 'M' for men's or 'W' for women's data.")
    }
    # Set year if NULL
    parsed_year <- ifelse(is.null(year), year(Sys.Date()), year)
    
    conf_info <- cbbdata::cbd_torvik_ratings(year = parsed_year) %>%
      distinct(team, conf)
    
    # Select the correct URL based on gender
    base_url <- if (gender == "M") {
      "https://barttorvik.com/trank.php"
    } else {
      "https://barttorvik.com/ncaaw/trank.php"
    }
    data_url <- glue("{base_url}?year={parsed_year}&csv=1")
    
    # Set column names
    data_names <- c("team", "adj_o", "adj_d", "barthag", "drop", "wins", "games", "efg", "def_efg", "ftr", "def_ftr",
                    "tov_rate", "def_tov_rate", "oreb_rate", "dreb_rate", "drop2", "two_pt_pct", "def_two_pt_pct", "three_pt_pct",
                    "def_three_pt_pct", "block_rate", "block_rate_allowed", "assist_rate", "def_assist_rate", "three_fg_rate",
                    "def_three_fg_rate", "adj_t", "drop3", "drop4", "drop5", "year", "drop6", "drop7", "drop8", "wab", "ft_pct", "def_ft_pct")
    
    # Read and clean data
    data <- read_csv(data_url, col_names = FALSE, show_col_types = FALSE) 
    print(data)
    data <- data %>%
      setNames(data_names) %>%
      left_join(conf_info, by = 'team') %>%
      mutate(losses = games - wins) %>%
      select(team, conf, games, wins, losses, adj_t, adj_o, adj_d, barthag, efg, def_efg, ftr, def_ftr,
             oreb_rate, dreb_rate, tov_rate, def_tov_rate, two_pt_pct, three_pt_pct, ft_pct, def_two_pt_pct,
             def_three_pt_pct, def_ft_pct, three_fg_rate, def_three_fg_rate, block_rate, block_rate_allowed,
             assist_rate, def_assist_rate, wab, year) %>%
      arrange(desc(barthag))
    
    return(data)
  })
}
# Combine data for years 2021-2024
years <- 2021:2024
w_combined_data <- bind_rows(lapply(years, function(y) cbd_torvik_team_factors(year = y, gender = "W")))

# View the combined dataset
#print(combined_data)

output_file <- "wbb_past_results.csv"
write_csv(w_combined_data, output_file)

# Output this season's results
current_women <- cbd_torvik_team_factors(2025, gender="W")
write_csv(current_women, "wbb_25.csv")

men_years <- 2008:2024
m_combined_data <- bind_rows(lapply(years, function(y) cbd_torvik_team_factors(year = y, gender = "M")))
write_csv(m_combined_data, "mbb_past_results.csv")
current_men <- cbd_torvik_team_factors(2025, "M")
write_csv(current_men, "mbb_25.csv")

