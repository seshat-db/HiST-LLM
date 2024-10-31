set datafile sep ','

se ytics nomirror
se y2tics 2000
se yr [0:0.64]   
se ke outside right center reverse Left spacing 1.5  

se xtics ("10,000 BCE" -10000, "8,000 BCE" -8000, "6,000 BCE" -6000, "4,000 BCE" -4000, "2,000 BCE" -2000, "0" 0, "2000 CE" 2000)
se xr [-10001:2001]

p   'new_ba_time.csv' u (($1+$2)/2):"gemini-1.5-flash-answer"                             w l lw 4 lc '0xa6cee3' t 'Gemini-1.5-flash'
rep 'new_ba_time.csv' u (($1+$2)/2):"gpt-3.5-turbo-0125-answer"                           w l lw 4 lc '0x1f78b4' t 'GPT-3.5-turbo'
rep 'new_ba_time.csv' u (($1+$2)/2):"gpt-4-turbo-2024-04-09-answer"                       w l lw 4 lc '0xb2df8a' t 'GPT-4-turbo'
rep 'new_ba_time.csv' u (($1+$2)/2):"gpt-4o-2024-05-13-answer"                            w l lw 4 lc '0x33a02c' t 'GPT-4o'
rep 'new_ba_time.csv' u (($1+$2)/2):"meta-llama/Llama-3-70b-chat-hf-answer"               w l lw 4 lc '0xfb9a99' t 'Llama-3-70B'
rep 'new_ba_time.csv' u (($1+$2)/2):"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-answer" w l lw 4 lc '0xe31a1c' t 'Llama-3.1-70B'
rep 'new_ba_time.csv' u (($1+$2)/2):"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-answer"  w l lw 4 lc '0xfdbf6f' t 'Llama-3.1-8B'
rep 'new_ba_time.csv' u (($1+$2)/2):"cnt" axes x1y2 w l lw 4 lc -1 t 'Number of data points'

se te post color eps solid size 5.5,2.4 "Helvetica" 12
se out 'ba_time.eps'
rep
se out
se te wxt

