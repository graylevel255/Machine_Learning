function plotClassData(category,modelData,data,var,k,nc,type)
predictedClassColumn = var+k+2;
if(strcmp(type,'nonparamknn'))
    predictedClassColumn = var+nc+3;
end
x_1 = modelData(:,1);
x_2 = modelData(:,2);
classes = modelData(:,predictedClassColumn);
dx_1 = data(:,1);
dx_2 = data(:,2);
orig_classes = data(:,var+1);

lyellow = [0.9961, 0.9961, 0.5859];
dyellow = [0.8438,0.6602,0];
lpink = [0.9961, 0.6445, 0.9141];
dpink = [0.4648,0.0078,0.4570];
lblue = [0.6758, 0.8555, 0.9961];
dblue = [0.0078,0.3438,0.4648];
lgreen = [0.2578, 0.9531, 0.7930];
dgreen = [0.0117,0.4453,0.0703];
 
 if(strcmp(category,'linear')||strcmp(category,'overlap'))
 figure;
 scatter(x_1(classes == 1),x_2(classes == 1),25,[0.9961, 0.9961, 0.5859],'filled') %yellow
 hold on;
 scatter(x_1(classes == 2),x_2(classes == 2),25,[0.9961, 0.6445, 0.9141],'filled') %pink
 hold on;
 scatter(x_1(classes == 3),x_2(classes == 3),25,[0.6758, 0.8555, 0.9961],'filled') %dark blue
 hold on;
 scatter(x_1(classes == 4),x_2(classes == 4),25,[0.2578, 0.9531, 0.7930],'filled') %green
 hold on;
 scatter(dx_1(orig_classes == 1),dx_2(orig_classes == 1),25,[0.8438,0.6602,0],'filled');
 hold on;
 scatter(dx_1(orig_classes == 2),dx_2(orig_classes == 2),25,[0.4648,0.0078,0.4570],'filled');
 hold on;
 scatter(dx_1(orig_classes == 3),dx_2(orig_classes == 3),25,[0.0078,0.3438,0.4648],'filled');
 hold on;
 scatter(dx_1(orig_classes == 4),dx_2(orig_classes == 4),25,[0.0117,0.4453,0.0703],'filled');
 
elseif(strcmp(category,'nonlinear'))
 figure;
 scatter(x_1(classes == 1),x_2(classes == 1),25,lyellow,'filled')
 hold on;
 scatter(x_1(classes == 2),x_2(classes == 2),25,lblue,'filled')
 hold on; 
 scatter(dx_1(orig_classes == 1),dx_2(orig_classes == 1),25,dyellow,'filled')
 hold on;
 scatter(dx_1(orig_classes == 2),dx_2(orig_classes == 2),25,dblue,'filled');
 hold on;
end
 xlabel('x1');
 ylabel('x2');
 title('Decision region prediction on Training data');
end