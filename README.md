# WikiCaps ETL Pipeline



## Metadata Columns

| ColumnId       	| Description                                                             	| Datatype 	|
|-------------------|---------------------------------------------------------------------------|-----------|
| wikicaps_id    	| ID (line number) of the row in the original WikiCaps Dataset __img_en__ 	| int      	|
| wikimedia_file 	| Wikimedia File ID of the Image associated with the Caption              	| str      	|
| caption        	| Caption of the Image                                                    	| str      	|
| image_path     	| Local path to the (downloaded) image                                    	| str      	|
| num_tok        	| Number of Tokens in the caption                                         	| int      	|
| num_sent       	| Number of Sentences in the caption                                      	| int      	|
| min_sent_len   	| Minimum number of Tokens in the Sentences of the caption                	| int      	|
| num_ne         	| Number of Named Entities in the caption                                 	| int      	|
| num_nouns      	| Number of Tokens with NOUN POS Tag **                                   	| int      	|
| num_propn      	| Number of Tokens with PROPN POS Tag **                                  	| int      	|
| num_conj       	| Number of Tokens with CONJ POS Tag **                                   	| int      	|
| num_verb       	| Number of Tokens with VERB POS Tag **                                   	| int      	|
| num_sym        	| Number of Tokens with SYM POS Tag **                                    	| int      	|
| num_num        	| Number of Tokens with NUM POS Tag **                                    	| int      	|
| num_adp        	| Number of Tokens with ADP POS Tag **                                    	| int      	|
| num_adj        	| Number of Tokens with ADJ POS Tag **                                    	| int      	|

** This column is only available if `config.input_data.pos_tag_stats == True`. [Click here](https://universaldependencies.org/docs/u/pos/) for a detailed description of the POS Tags
