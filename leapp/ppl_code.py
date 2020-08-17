import pymc3 as pm
import theano.tensor as tt
def trace():
     with pm.Model() as model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), educ*0.2477+6.5963, educ*0.2059+7.295), sigma=tt.switch(tt.eq(eastwest, 0), 1.7444, 1.7047))
        income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*154.9222+happiness*75.795+144.8802, educ*326.6378+happiness*116.0607+-279.4209), tt.switch(tt.eq(sex, 0), educ*220.2771+happiness*-0.0931+594.0865, educ*384.5272+happiness*184.258+-380.217)), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 636.2516, 956.2709), tt.switch(tt.eq(sex, 0), 845.3302, 1437.4018)))
        age = pm.Normal('age', mu=tt.switch(tt.eq(eastwest, 0), educ*-4.7345+income*0.0+70.8893, educ*-5.3423+income*0.0025+65.1793), sigma=tt.switch(tt.eq(eastwest, 0), 16.4303, 16.2479))
        health = pm.Normal('health', mu=age*-0.0161+educ*0.0921+income*0.0001+happiness*0.214+2.3658, sigma=0.8404)
        return pm.sample(1000)