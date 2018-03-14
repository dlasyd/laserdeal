# Model description

## History
1) Loss function uses a separate nn to move stop losses based on new price information.

2) Loss function uses the same nn that it is training to move stop losses

3) NN has an output "in the middle" to start trading and then outputs corrections to stop loss and take profit

## Model description
### Output

|index   	|  description 	        |
|---	    |---	                |
|0	        |buy  	                |
|1   	    |stop loss              |
|2   	    |take profit   	        |
|3  	    |sell  	                |
|4   	    |stop loss   	        |
|5   	    |take profit   	        |
|6   	    |buy sl correction   	|
|7   	    |buy tp correction   	|
|8          |buy end deal           |
|9   	    |sell sl correction   	|
|10   	    |sell tp correction   	|
|11         |sell end deal          |

There are 4 blocks of 3. Only one of them makes sense in any given moment.
On the first output it is decided whether to buy, or to sell or to do nothing. Max(sigmoid(buy),sigmoid(sell)), both>0
After that values from appropriate inside-deal section are used.

TODO add flags to change sl and tp