# Neural-network
The provided data describe fragments of Starcraft games played between two players on standard 1v1 maps. Each row in the data files describes approximately 1 minute of a game. The first column in the training data contains game IDs, and the second column is the decision - information about the game-winner. The third column indicates the game type - which of the three in-game races are fighting. The remaining columns contain characteristics describing the situation in the game map. Their meaning is reflected by their names.

The data tables are provided as two CSV files with the ',' (coma) separator sign. Both files (training and test sets) have the same format but the GameID and decision columns are missing from the test data.

The evaluation metric will be AUC.
