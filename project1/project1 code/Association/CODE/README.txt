
Steps to Run the code

a) Install python3

b) unzip file and change directory to 'CODE'

c) run the command "python3 apriori.py"

d) Enter the file name 

e) Enter the support Value

f) Enter the Confidence Value

g) Enter the template number and relevant Template parameters

h) The rules and the corresponding rule count are generated


Sample Output

Viveks-MacBook-Air:Code viveknagaraju$ python3 apriori.py 
Enter the name of the input file: associationruletestdata.txt
Enter the Minimum Support Count: 50
Support is set to be 50%
length-1 freq item sets:  109
length-2 freq item sets: 63
length-3 freq item sets: 2
length-4 freq item sets: 0
length of itemset freq 174
Enter confidence percentage: 70
rule count ---- 117
Enter the template number: 2
Enter First parameter for template 2: head
Enter Second parameter for template 2: 1
the final set of template rules are:
{'G72_UP->G82_DOWN', 'G13_DOWN->G6_UP', 'G88_DOWN->G8_UP', 'G6_UP->G13_DOWN', 'G88_DOWN->G87_UP', 'G87_UP->G28_DOWN', 'G47_UP->G28_DOWN', 'G28_DOWN->G59_UP', 'G13_DOWN->G28_DOWN', 'G97_DOWN->G82_DOWN', 'G32_DOWN->G6_UP', 'G28_DOWN->G2_DOWN', 'G38_DOWN->G47_UP', 'G10_DOWN->G70_DOWN', 'G10_DOWN->G38_DOWN', 'G38_DOWN->G1_UP', 'G96_DOWN->G59_UP,G72_UP', 'G41_DOWN->G28_DOWN', 'G32_DOWN->G59_UP', 'G47_UP->G38_DOWN', 'G94_UP->G10_DOWN', 'G59_UP,G72_UP->G82_DOWN', 'G13_DOWN->G72_UP', 'G38_DOWN->G28_DOWN', 'G88_DOWN->G54_UP', 'G59_UP->G96_DOWN', 'G82_DOWN->G59_UP', 'G72_UP,G96_DOWN->G59_UP', 'G59_UP->G72_UP', 'G82_DOWN->G97_DOWN', 'G8_UP->G88_DOWN', 'G38_DOWN->G59_UP', 'G70_DOWN->G1_UP', 'G88_DOWN->G10_DOWN', 'G87_UP->G88_DOWN', 'G59_UP->G82_DOWN', 'G70_DOWN->G10_DOWN', 'G28_DOWN->G32_DOWN', 'G96_DOWN->G82_DOWN', 'G6_UP->G28_DOWN', 'G28_DOWN->G87_UP', 'G10_DOWN->G94_UP', 'G24_DOWN->G88_DOWN', 'G88_DOWN->G28_DOWN', 'G1_UP->G59_UP', 'G87_UP->G59_UP', 'G59_UP,G72_UP->G96_DOWN', 'G1_UP->G38_DOWN', 'G72_UP->G59_UP', 'G88_DOWN->G38_DOWN', 'G97_DOWN->G72_UP', 'G72_UP->G1_UP', 'G59_UP->G13_DOWN', 'G72_UP->G96_DOWN', 'G38_DOWN->G10_DOWN', 'G13_DOWN->G59_UP', 'G70_DOWN->G38_DOWN', 'G6_UP->G38_DOWN', 'G10_DOWN->G47_UP', 'G67_UP->G38_DOWN', 'G88_DOWN->G41_DOWN', 'G10_DOWN->G28_DOWN', 'G54_UP->G24_DOWN', 'G28_DOWN->G6_UP', 'G82_DOWN->G72_UP', 'G96_DOWN->G72_UP', 'G1_UP->G54_UP', 'G28_DOWN->G88_DOWN', 'G41_DOWN->G38_DOWN', 'G88_DOWN->G24_DOWN', 'G2_DOWN->G38_DOWN', 'G41_DOWN->G88_DOWN', 'G82_DOWN->G59_UP,G72_UP', 'G1_UP->G70_DOWN', 'G59_UP,G96_DOWN->G72_UP', 'G6_UP->G32_DOWN', 'G32_DOWN->G28_DOWN', 'G28_DOWN->G47_UP', 'G72_UP,G82_DOWN->G59_UP', 'G47_UP->G10_DOWN', 'G82_DOWN->G13_DOWN', 'G82_DOWN->G96_DOWN', 'G52_DOWN->G28_DOWN', 'G1_UP->G10_DOWN', 'G1_UP->G67_UP', 'G10_DOWN->G88_DOWN', 'G28_DOWN->G52_DOWN', 'G72_UP->G59_UP,G82_DOWN', 'G88_DOWN->G59_UP', 'G94_UP->G38_DOWN', 'G38_DOWN->G52_DOWN', 'G28_DOWN->G10_DOWN', 'G54_UP->G1_UP', 'G28_DOWN->G41_DOWN', 'G28_DOWN->G38_DOWN', 'G6_UP->G59_UP', 'G91_UP->G38_DOWN', 'G10_DOWN->G1_UP', 'G1_UP->G72_UP', 'G32_DOWN->G38_DOWN', 'G2_DOWN->G28_DOWN', 'G38_DOWN->G32_DOWN', 'G65_DOWN->G38_DOWN', 'G52_DOWN->G38_DOWN', 'G72_UP->G13_DOWN', 'G59_UP,G82_DOWN->G72_UP', 'G54_UP->G88_DOWN', 'G13_DOWN->G82_DOWN', 'G96_DOWN->G59_UP', 'G32_DOWN->G72_UP', 'G38_DOWN->G91_UP', 'G28_DOWN->G13_DOWN', 'G38_DOWN->G70_DOWN', 'G10_DOWN->G59_UP', 'G59_UP->G88_DOWN', 'G24_DOWN->G54_UP', 'G67_UP->G1_UP'}
the count of the template rules are ... 117
Viveks-MacBook-Air:Code viveknagaraju$ 