function predictData = NP_find_min_Rmax(predictData,var,nc)
pd = size(predictData,1);
initializeMinimumRmax = zeros(pd,1);
initializePredictedClass = zeros(pd,1);
predictData = [predictData,initializeMinimumRmax,initializePredictedClass];

m = var+1;
min = 9999999999;
predictedClass = 0;
finalIndex = var+1+nc+1;

for i = 1:pd
   for j = 1:nc
       if predictData(i,m+j)<min
           min = predictData(i,m+j);
           predictedClass = j;
       end
   end
   predictData(i,finalIndex) = min;
   predictData(i,finalIndex+1) = predictedClass;
   min = 9999999999;
end

end