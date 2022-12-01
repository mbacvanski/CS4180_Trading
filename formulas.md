$$
\begin{align}
\text{State}&:S\in\begin{bmatrix}
[\text{Close prices}]\\
[\text{Open price}]\\
[\text{Day high prices}]\\
[\text{Day low prices}]\\
[\text{Adjusted close prices}]\\
[\text{Volume}]\\
[\text{Position history}]\\
\end{bmatrix}^T\\
\text{Actions}&:\begin{cases}
\text{buy}\\
\text{hold}\\
\text{sell}
\end{cases}\\
\text{Rewards}&:many options\\
\text{Transitions}&:\begin{bmatrix}
[\text{Close prices[1:] + next close price}]\\
[\text{Open prices[1:] + next open price}]\\
[\text{Day high prices[1:] + next day high price}]\\
[\text{Day low prices[1:] + next day low price}]\\
[\text{Adjusted close prices[1:] + next day adj. close price}]\\
[\text{Volume[1:] + next day volume}]\\
[\text{Position history[1:] + next position}]\\
\end{bmatrix}^T\\
\text{Position}&:\begin{cases}
-1&\text{if SHORT}\\
0&\text{if FLAT}\\
1&\text{if LONG}\\
\end{cases}\\
\text{Position}&\gets\begin{cases}
\text{SHORT if SHORT and HOLD}\\
\text{SHORT if SHORT and SELL}\\
\text{SHORT if FLAT and SELL}\\
\\
\text{FLAT if FLAT and HOLD}\\
\text{FLAT if SHORT and BUY}\\
\text{FLAT if LONG and SELL}\\
\\
\text{LONG if LONG and HOLD}\\
\text{LONG if LONG and BUY}\\
\text{LONG if FLAT and BUY}\\
\end{cases}
\end{align}
$$






