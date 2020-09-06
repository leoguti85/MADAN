

disney_dict = {'attributes': ['MinPricePrivateSeller','Avg_Rating'], 'sigma': 0.32, 'num_net': 1}


books_dict = {'attributes': ['Min_Helpful','MinPriceUsedItem', 'Min_Votes', 'Rating_3_Ratio', 'Rating_span', 'Sales_Rank', 'Review_frequency', 'Helpful_votes_ratio', 'Avg_Votes', 'Rating_5_Ratio',  'Max_Votes',  'Max_Helpful',  'Rating_2_Ratio', 'Amazon_price', 'MinPricePrivateSeller','Avg_Rating', 'Rating_4_Ratio', 'Rating_1_Ratio', 'Number_of_reviews', 'Avg_Helpful'], 
			  'sigma': 0.15, 'num_net': 3}


enron_dict = {'attributes': ['AverageContentForwardingCount','OtherMailsBcc','AverageDifferentSymbolsSubject','OtherMailsCc','EnronMailsTo'], 
			  'sigma': 0.1, 'num_net': 4}

params_db   = {'disney': disney_dict, 'books': books_dict, 'enron': enron_dict}

#enron_attrib = ['AverageContentReplyCount',  'AverageContentLength',  'EnronMailsBcc',  'AverageContentForwardingCount', 'AverageRangeBetween2Mails', 'OtherMailsBcc',  'AverageNumberBcc', 'AverageDifferentSymbolsContent',  'AverageDifferentSymbolsSubject',  'OtherMailsTo',  'DifferentCharsetsCount',  'DifferntCosCount',  'OtherMailsCc',  'AverageNumberTo',  'AverageNumberCc', 'EnronMailsTo', 'DifferentEncodingsCount', 'MimeVersionsCount']  