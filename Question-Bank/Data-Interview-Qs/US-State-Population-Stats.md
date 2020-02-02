## Question
Given the following datasets:
* [US births/deaths by state](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHLx3MSbwZrt8DwLPWUgrfy-2FWYTmmmnVSeu5gKS69ghQpghilpIGfmsHCKyHuT6I8QlIrPzZIKhgjLpm7cmzI1vf2IUrHxdvJXnDnsQflvY8d8-2F-2BBQ6p7WhEN4w0muQnYjc2bJp-2Bd5cQ5RaPrSd-2FEdkCM-3D_8c6kLYfeKFgEvI6pydPvKCo5RIOwGXukDLGeEAsdKQMP3EhJoFKQW47BsRocF7VqPrJ7NAtDSAOZBUPJ9bEm9QccHpL-2BU-2BijxbDNyy3wwFnIrJEINQRvQ-2FC-2BYfA-2BONjbyBBEiHy-2FJW-2FPy7gjbo2Cbh63GHVWFuhaql-2FnA-2BosgE5h6muARrqwnuIM8lCdM5hHvThyPNAsOGBtJbDbPXcvPYkCWSJVMazkl4-2Bd7eGzDwU-3D)
* [U.S. population by state](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHLx3MSbwZrt8DwLPWUgrfy-2FWYTmmmnVSeu5gKS69ghQpghilpIGfmsHCKyHuT6I8QlIrPzZIKhgjLpm7cmzI1vf2IUrHxdvJXnDnsQflvY8d8-2F-2BBQ6p7WhEN4w0muQnYjc2bJp-2Bd5cQ5RaPrSd-2FEdkCM-3D_8c6kLYfeKFgEvI6pydPvKCo5RIOwGXukDLGeEAsdKQMP3EhJoFKQW47BsRocF7Vq8eH0ZTm-2BbgK1kCOBgM6YWxwcaLvFHm0Dk18FeVAzVtuSX5oF4J0uZ79PxPdVQ3yU1VgwvLDpQwksawBwEee7-2BUMC5oPL7VzZh8YaXy1Rw8RnhguAZPXlmvvwPHGdx51cyJ-2B7Md70vbUWYJuRVvtHZhvfggvEFoR4jJUkYfSkRY4-3D)

**Answer the following questions**:
1. Calculate the birth rate and death rate for each state. For the purposes of this question you can define the rate as the # of births/deaths divided by the total population.
2. Create a new column appended to the first dataset with the net population change (births-deaths) by state.
3. Using the column from (2), project out the future population by state in 5 years, assuming the population change remains the same (on an absolute/n-count basis). You can assume each state's net transfer in/out rate from folks moving is 0 for this question's sake.

*For example, if CA's population is 30M and the birth-death rate is 150k, then the projected population in 5 years is just 30,000,000 + 150,000*5 = 30.75M.*

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1YNOgngHuDQwsSTckk1QAzpiIvrHu2-qi) to view this solution in an interactive Colab (Jupyter) notebook.

Some good follow ups to this would be to consider other factors that influence state population rates (such as net adds in/out from moves), what variables one might use to predict aside from raw population growth (e.g. economic growth, job creation, age distribution of population, etc). -->
