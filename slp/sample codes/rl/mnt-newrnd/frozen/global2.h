/* types of events 0=termination; 1=arrive; 2=produce;3=repair 4=maint */

int NUMBER_OF_EVENTS=5; 


double LENGTH=1000000;

double BIG   =1000001; /* has to be bigger than LENGTH */

# define PCMAX 100 /* max value for production count */
# define UPPER_LIMIT 3 /* S value for buffer */
# define LOWER_LIMIT 2 /* s value for buffer */
 

/* the following are the paramters of gamma distribution */
/* for PROD=production time, FAIL=time between failures, */
/* and REPAIR= repair time */
int N_PROD=8;
double LAMBDA_PROD=0.8;

int N_FAIL=8;
double LAMBDA_FAIL=0.08;

int N_REPAIR=2;
double LAMBDA_REPAIR=0.01;

double MU=0.1; /* exponential rate of arrival of demands */

/* the following parameters are for the unif dist. for maintenance */

double MAINT_MIN=5;
double MAINT_MAX=20;


double TNOW;

double PROFIT=1;

double MAINT_COST=2;

double REPAIR_COST=5;

int T1=5; /* denotes the threshold for maintenance when buffer=1 */
int T2=6; /* denotes the threshold for maintenance when buffer=2 */
int T3=7; /* denotes the threshold for maintenance when buffer=3 */

long REPLICATE;     

