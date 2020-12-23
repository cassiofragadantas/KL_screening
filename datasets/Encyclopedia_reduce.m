% This scripts uses the Grolier Encyclopedia word count dataset available at:
% https://cs.nyu.edu/~roweis/data.html
%
% Here we reduce it to the set of ~7500 most frequent words

load('grolier15276.mat')

% Sort words according to frequency
[~, idx] = sort(sum(grolier,2));

ordered_idx = sort(idx(7777:end)); %in order not to lose the alphabetic order
grolier = grolier(ordered_idx,:);
words = words(ordered_idx);

save('grolier7500.mat','grolier','words')