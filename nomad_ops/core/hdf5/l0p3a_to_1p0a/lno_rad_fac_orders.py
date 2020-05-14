# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:37:15 2020

@author: iant

LNO RADIANCE FACTOR ORDERS

"""

#diffraction order: [is solar line good enough for fullscan fit?,
#"search for solar or molecular lines or none"
#"best molecule (if any)"]

#[nadir mean signal cutoff px 160:240, minimum signal for absorption, n stds for absorption, n stds for reference spectrum absorption]
#can't do 197+ due to no solar spectrum

rad_fact_orders_dict = {
115:{"solar_line":True, "trans_solar":0.8, "nu_range":[2585., 2587.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
116:{"solar_line":False, "solar_molecular":""},
117:{"solar_line":False, "solar_molecular":""},
118:{"solar_line":True, "trans_solar":0.95, "nu_range":[2669., 2671.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
119:{"solar_line":True, "trans_solar":0.96, "nu_range":[2690., 2692.], "solar_molecular":"solar", "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},

120:{"solar_line":True, "trans_solar":0.88, "nu_range":[2714., 2716.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
121:{"solar_line":True, "trans_solar":0.95, "nu_range":[2732., 2734.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
122:{"solar_line":True, "trans_solar":0.97, "nu_range":[2753., 2755.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
123:{"solar_line":False, "solar_molecular":""},
124:{"solar_line":False, "solar_molecular":""},
125:{"solar_line":False, "solar_molecular":""},
126:{"solar_line":True, "trans_solar":0.95, "nu_range":[2837., 2839.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
127:{"solar_line":False, "solar_molecular":""},
128:{"solar_line":False, "solar_molecular":""},
129:{"solar_line":False, "solar_molecular":""},

130:{"solar_line":True, "trans_solar":0.89, "nu_range":[2943., 2945.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
131:{"solar_line":True, "trans_solar":0.97, "nu_range":[2959., 2961.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
132:{"solar_line":True, "trans_solar":0.95, "nu_range":[2981., 2983.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
133:{"solar_line":True, "trans_solar":0.85, "nu_range":[3011., 3013.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
134:{"solar_line":True, "trans_solar":0.92, "nu_range":[3013., 3014.5], "solar_molecular":"solar", "mean_sig":1.5, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
135:{"solar_line":True, "trans_solar":0.97, "nu_range":[3043., 3045.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
136:{"solar_line":False, "solar_molecular":""},
137:{"solar_line":True, "trans_solar":0.95, "nu_range":[3083., 3085.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
138:{"solar_line":False, "solar_molecular":""},
139:{"solar_line":False, "solar_molecular":""},

140:{"solar_line":True, "trans_solar":0.96, "nu_range":[3153., 3155.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
141:{"solar_line":True, "trans_solar":0.9, "nu_range":[3172., 3174.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
142:{"solar_line":True, "trans_solar":0.9, "nu_range":[3208., 3210.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
143:{"solar_line":False, "solar_molecular":""},
144:{"solar_line":False, "solar_molecular":""},
145:{"solar_line":False, "solar_molecular":""},
146:{"solar_line":True, "trans_solar":0.95, "nu_range":[3288., 3290.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
147:{"solar_line":False, "solar_molecular":""},
148:{"solar_line":False, "solar_molecular":""},
149:{"solar_line":False, "solar_molecular":""},

150:{"solar_line":False, "solar_molecular":""},
151:{"solar_line":True, "trans_solar":0.94, "nu_range":[3413., 3415.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
152:{"solar_line":True, "trans_solar":0.96, "nu_range":[3429., 3431.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
153:{"solar_line":True, "trans_solar":0.95, "nu_range":[3462., 3464.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
154:{"solar_line":True, "trans_solar":0.95, "nu_range":[3483., 3484.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
155:{"solar_line":True, "trans_solar":0.97, "nu_range":[3496., 3497.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
156:{"solar_line":True, "trans_solar":0.95, "nu_range":[3519., 3521.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
157:{"solar_line":False, "solar_molecular":"molecular", "molecule":"CO2"},
158:{"solar_line":True, "trans_solar":0.98, "nu_range":[3568., 3570.], "solar_molecular":"molecular", "molecule":"CO2", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
159:{"solar_line":False, "solar_molecular":"molecular", "molecule":"CO2"},

160:{"solar_line":True, "trans_solar":0.97, "nu_range":[3614., 3616.], "solar_molecular":"molecular", "molecule":"CO2", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
161:{"solar_line":False, "solar_molecular":"molecular", "molecule":"CO2"},
162:{"solar_line":True, "trans_solar":0.9, "nu_range":[3650., 3652.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
163:{"solar_line":True, "trans_solar":0.92, "nu_range":[3687., 3689.], "solar_molecular":"molecular", "molecule":"CO2", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
164:{"solar_line":True, "trans_solar":0.93, "nu_range":[3693., 3695.], "solar_molecular":"molecular", "molecule":"CO2", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
165:{"solar_line":False, "solar_molecular":"molecular", "molecule":"CO2"},
166:{"solar_line":True, "trans_solar":0.91, "nu_range":[3749., 3751.], "solar_molecular":"molecular", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
167:{"solar_line":True, "trans_solar":0.89, "nu_range":[3766., 3768.], "solar_molecular":"molecular", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
168:{"solar_line":True, "trans_solar":0.78, "nu_range":[3787., 3789.], "solar_molecular":"both", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
169:{"solar_line":True, "trans_solar":0.9, "nu_range":[3812., 3814.], "solar_molecular":"both", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},

170:{"solar_line":True, "trans_solar":0.97, "nu_range":[3843., 3845.], "solar_molecular":"molecular", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
171:{"solar_line":True, "trans_solar":0.85, "nu_range":[3866., 3867.], "solar_molecular":"molecular", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
172:{"solar_line":True, "trans_solar":0.92, "nu_range":[3894., 3896.], "solar_molecular":"molecular", "molecule":"H2O", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
173:{"solar_line":True, "trans_solar":0.93, "nu_range":[3905., 3906.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
174:{"solar_line":True, "trans_solar":0.77, "nu_range":[3933., 3935.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
175:{"solar_line":False, "solar_molecular":""},
176:{"solar_line":False, "solar_molecular":""},
177:{"solar_line":False, "solar_molecular":""},
178:{"solar_line":True, "trans_solar":0.77, "nu_range":[4020., 4022.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
179:{"solar_line":True, "trans_solar":0.91, "nu_range":[4042., 4044.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},

180:{"solar_line":True, "trans_solar":0.89, "nu_range":[4068., 4070.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
181:{"solar_line":True, "trans_solar":0.945, "nu_range":[4087., 4089.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
182:{"solar_line":True, "trans_solar":0.955, "nu_range":[4100., 4102.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
183:{"solar_line":False, "solar_molecular":""},
184:{"solar_line":True, "trans_solar":0.97, "nu_range":[4156., 4158.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
185:{"solar_line":True, "trans_solar":0.97, "nu_range":[4169., 4171.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
186:{"solar_line":True, "trans_solar":0.93, "nu_range":[4189., 4191.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
187:{"solar_line":True, "trans_solar":0.96, "nu_range":[4218., 4220.], "solar_molecular":"solar", "molecule":"CO", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
188:{"solar_line":True, "trans_solar":0.92, "nu_range":[4250., 4252.], "solar_molecular":"molecular", "molecule":"CO", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
189:{"solar_line":True, "trans_solar":0.86, "nu_range":[4275., 4277.], "solar_molecular":"molecular", "molecule":"CO", "mean_sig":4.0, "min_sig":2.0, "stds_sig":1.0, "stds_ref":1.0},

190:{"solar_line":True, "trans_solar":0.9, "nu_range":[4281., 4283.], "solar_molecular":"molecular", "molecule":"CO", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
191:{"solar_line":True, "trans_solar":0.96, "nu_range":[4319., 4321.], "solar_molecular":"molecular", "molecule":"CO", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
192:{"solar_line":False, "solar_molecular":"molecular", "molecule":"CO"},
193:{"solar_line":True, "trans_solar":0.94, "nu_range":[4364., 4365.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":6.0},
194:{"solar_line":True, "trans_solar":0.79, "nu_range":[4382., 4384.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":0.45, "stds_ref":4.0},
195:{"solar_line":True, "trans_solar":0.932, "nu_range":[4413., 4414.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":2.0, "stds_ref":2.0},
196:{"solar_line":True, "trans_solar":0.9, "nu_range":[4421., 4423.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":0.45, "stds_ref":4.0},
197:{"solar_line":True, "trans_solar":0.88, "nu_range":[4448., 4450.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":0.45, "stds_ref":4.0},
198:{"solar_line":True, "trans_solar":0.88, "nu_range":[4466., 4468.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":0.45, "stds_ref":4.0},
199:{"solar_line":True, "trans_solar":0.9, "nu_range":[4491.2, 4493.], "solar_molecular":"solar", "mean_sig":4.0, "min_sig":2.0, "stds_sig":0.45, "stds_ref":4.0},
}

