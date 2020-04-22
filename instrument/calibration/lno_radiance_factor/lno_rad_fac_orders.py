# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:37:15 2020

@author: iant

LNO RADIANCE FACTOR ORDERS

"""

#diffraction order: ["search for solar or molecular lines or none", "best molecule (if any)"]
#[nadir mean signal cutoff, minimum signal for absorption, n stds for absorption, n stds for reference spectrum absorption]
#can't do 197+ due to no solar spectrum

BEST_ABSORPTION_DICT = {
115:["Solar", ""],
116:["", ""],
117:["", ""],
118:["Solar", ""],
119:["Solar", ""],

120:["Solar", ""],
121:["Solar", ""],
122:["Solar", ""],
123:["", ""],
124:["", ""],
125:["", ""],
126:["Solar", ""],
127:["", ""],
128:["", ""],
129:["", ""],

130:["Solar", ""],
131:["Solar", ""],
132:["Solar", ""],
133:["Solar", ""],
134:["", ""],
135:["Solar", ""],
136:["", ""],
137:["Solar", ""],
138:["Solar", ""],
139:["Solar", ""],

140:["Solar", ""],
141:["Solar", ""],
142:["Solar", ""],
143:["", ""],
144:["Solar", ""],
145:["Solar", ""],
146:["Solar", ""],
147:["Solar", ""],
148:["Solar", ""],
149:["Solar", ""],

150:["Solar", ""],
151:["Solar", ""],
152:["Solar", ""],
153:["Solar", ""],
154:["Solar", ""],
155:["Solar", ""],
156:["Solar", ""],
157:["", ""],
158:["Molecular", "CO2"],
159:["Molecular", "CO2"],

160:["Molecular", "CO2", [4.0, 2.0, 2.0, 2.0]],
161:["Molecular", "CO2"],
162:["Solar", "", [4.0, 2.0, 2.0, 2.0]],
163:["Molecular", "CO2", [4.0, 2.0, 2.0, 2.0]],
164:["Molecular", "CO2"],
165:["Molecular", "CO2"],
166:["Molecular", "H2O"],
167:["Molecular", "H2O", [4.0, 2.0, 2.0, 2.0]],
168:["Molecular", "H2O", [4.0, 2.0, 2.0, 2.0]],
169:["Molecular", "H2O", [4.0, 2.0, 2.0, 2.0]],

170:["Molecular", "H2O"],
171:["Molecular", "H2O"],
172:["Molecular", "H2O"],
173:["Solar", ""],
174:["Solar", ""],
175:["", ""],
176:["Solar", ""],
177:["Solar", ""],
178:["Solar", ""],
179:["Solar", ""],

180:["Solar", ""],
181:["Solar", ""],
182:["Solar", ""],
183:["Solar", ""],
184:["Solar", ""],
185:["Solar", ""],
186:["Solar", ""],
187:["Molecular", "CO"],
188:["Molecular", "CO"],
189:["Molecular", "CO", [4.0, 2.0, 1.0, 1.0]],

190:["Molecular", "CO"],
191:["Molecular", "CO"],
192:["Solar", ""],
193:["Solar", ""],
194:["Solar", "", [4.0, 2.0, 0.45, 4.0]],
195:["Solar", ""],
196:["Solar", "", [4.0, 2.0, 0.45, 4.0]],
197:["", ""],
198:["", ""],
199:["", ""],
}

#fill in generic values for other orders
for key, value in BEST_ABSORPTION_DICT.items():
    if len(value)==2:
        BEST_ABSORPTION_DICT[key].append([4.0, 2.0, 2.0, 2.0])