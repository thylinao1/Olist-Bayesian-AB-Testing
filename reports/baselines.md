# Classical baseline results

Treatment: free-shipping eligibility (post-cutover & subtotal >= R$ 150)
Cutover week: **2018-01-15**

## Two-proportion z-test on on-time delivery

```
  treated  : 10,784/12,279  (87.82%)
  control  : 76,652/84,998  (90.18%)
  diff     : -2.356 pp,  95% CI [-2.968, -1.744] pp
  z        : -8.094
  p-value  : 0.0000
```

## Welch t-test on per-customer repeat revenue (180-day window)

```
  mean(treated): R$     5.54
  mean(control): R$     3.63
  diff         : R$ +1.90, 95% CI [+0.44, +3.37]
  t            : +2.542,  df=12561.2
  p-value      : 0.0110
```

## Mann-Whitney U on per-customer repeat revenue

```
  median(treated): R$     0.00
  median(control): R$     0.00
  U statistic    : 482,519,994
  p-value        : 0.0002
```

## Chi-square test of independence: review score x treatment

```
review_score     1     2     3      4      5
treatment                                   
0             7589  2456  6862  16423  48959
1             1561   415   938   2136   6853
chi^2  : 201.251, df=4
p-value: 0.0000
```

Mann-Whitney U on review score: U=472,044,377, p=0.0000
