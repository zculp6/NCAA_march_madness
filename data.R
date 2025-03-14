#devtools::install_github("andreweatherman/cbbdata") # free package in R
library("cbbdata")
cbbdata::cbd_login()
cbb_25 <- cbbdata::cbd_torvik_team_factors(year=2025, no_bias = T)
write.csv(cbb_25, "cbb_25.csv", row.names=FALSE)

install.packages("RSelenium")
library(RSelenium)

# Start RSelenium with Chrome browser (you can use Firefox or other supported browsers)
driver <- rsDriver(browser = "chrome", port = 4445L)
remote_driver <- driver$client

# Navigate to the Bart Torvik Women's Basketball T-Rank page
remote_driver$navigate("https://barttorvik.com/ncaaw/trank.php")

# Give the page some time to load (adjust the time if necessary)
Sys.sleep(5)  # Wait 5 seconds for the page to fully load

# Get the page source after the JavaScript has rendered the table
page_source <- remote_driver$getPageSource()[[1]]

# You can now use rvest to scrape the table
library(rvest)

# Read the page source into rvest
webpage <- read_html(page_source)

# Extract the tables (if there are multiple, we select the correct one)
tables <- html_table(webpage, fill = TRUE)

# Check the extracted tables (there might be more than one, depending on the page structure)
str(tables)

# Save the desired table (the first one in this case) as a CSV
wbb_data <- tables[[1]]  # Assuming the first table is the one you're looking for
write.csv(wbb_data, "barttorvik_wbb_2025.csv", row.names = FALSE)

# Close the RSelenium session when done
remote_driver$close()




