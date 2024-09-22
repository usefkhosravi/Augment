# Augment
The source code of my paper: "Improving deep learning in arrhythmia Detection: The application of modular quality and quantity controllers in data augmentation"

address:
https://www.sciencedirect.com/science/article/abs/pii/S1746809423013733


In this research, an innovative method to improve the data augmentation of one-dimensional signals has been presented.
To better recognize and classify data with deep convolutional neural networks.
6 data augmentation methods has been used: 1- Scaling 2- Time warping 3- Magnitude warping 4- Frequency component warping 5- Gaussian (A state such as jittering augmentation but with special selection and deviation according to GDO article)  6- LSTM  7- GAN . 
 The innovation of research is determining the appropriate intensity and share and volume for each of data augmentation methods, And even dedicated for each class sorting augmented data according to several quality mesurements and Selecting a part of them with appropriate statistical dispersion.

The MITBIH database, which contains one-dimensional signals of different classes of cardiac arrhythmias, has been tested. Also, other data that have been reviewed in my master's thesis have also been included.



Tips:

1- Codes that have the term 0CNN in their name; They choose the superior convolutional neural network architecture. The architectures include: 1-simple architecture 2-simple deep architecture 3-ResNext architecture is examined with different depths. Heartbeats are extracted from each long-term signal and classification and labeling are done, and the label and class in Long-term signal fragments that are repeated more often are the final label. The classification accuracy of each class is obtained and recorded from the results of this code to be used for the beta index in step 4.

2- The code that has the phrase 1CNN in the middle of its name generates a lot of data with 7 data augmentation methods, each with different intensities and stores it in the AugEvl folder. In this code, 5 qualitative dispersion indices for each sample resulting from data augmentation are calculated and added to the end of the signal of that sample. From the last to the first, these indicators are 1- borderline 2- non-borderline 3- ratio of distance with classmates to distance with non-classmates 4- variance of distance with 5 nearest neighbors 5- distance with classmates.

3- The code that has the words 2CNN or 3CNN in the middle of its name determines the appropriate dispersion for selecting the necessary artificial samples. The ones that were stored in the AugEvl folder. Artificial samples are sorted each time based on one of 5 quality indicators and are selected with one of 5 distributions: 1-ordinal, 2-linear, 3-beta, 4-exponential, 5-uniform, and are used for neural network training. That is, 25 selection modes are checked for the samples obtained from each data augmentation method. Of course, in this code, 3 qualitative-distributive modes are already considered, and the superior qualitative-distributive index is determined for selecting samples of each data augmentation method. At this stage, the best classification accuracy of the trained neural network is also recorded with the data obtained from each data augmentation method.

4- The code that has the phrase 4CNN in the middle of its name performs different volumes of data augmentation with X coefficients for the total volume of data augmentation and alpha for the volume of each data augmentation method and beta for each class to obtain the superior data augmentation method. At this stage, Cubuk and alphatrim methods have also been examined.

5- The folders that have "-" in their last names show that the results obtained with the method presented by me were weaker than the recent researches.

6-The random and borderline difference in the code and title of the files refers to the method of selecting the base sample to apply the data augmentation method on it. In borderline mode, more chances are given to borderline samples and close to opposite classes for data augmentation, which is according to Jedio's article. But because in the current method, the samples are quality measured, the huge amount of data from all the samples are selected with the same number of production and quality measurement and with better dispersion.

-----------------------------------------------------------------------


In Persian:

پروژه حاضر پیاده سازی و کد تحقیقاتی مقاله با عنوان:
 Improving deep learning in arrhythmia Detection: The application of modular quality and quantity controllers in data augmentation 
به آدرس:
https://www.sciencedirect.com/science/article/abs/pii/S1746809423013733
است.
در این پژوهش روشی ابداعی در جهت بهبود داده افزایی سیگنال های تک بعدی ارائه شده است.
 تا تشخیص و کلاس بندی داده ها با شبکه های عصبی عمیق کانولوشنی بهتر انجام شود.
از 6 روش داده افزایی 1-تغییر مقیاس، 2-انحراف زمان و 3-انحراف دامنه و4-انحراف مولفه های فرکانسی 5-گوسی (جیترینگ با انحراف به روش مقاله GDO) داده افزایی با 6-شبکه های LSTM و 7- GAN استفاده شده است.
 نوآوری تحقیق تعیین شدت ها و  سهم و حجم مناسب برای هر یک از آن ها و اختصاصی شده برای هر کلاس و کیفیت سنجی و انتخاب با پراکندگی مناسب است.

پایگاه داده MITBIH که سیگنال های تک بعدی کلاس های مختلف آریتمی های قلب است مورد آزمایش قرار گرفته است. همچنین داده های دیگری که در پایان نامه کارشناسی ارشد بنده بررسی شده اند نیز قرار داده شده اند.


1-کدهایی که عبارت 0CNN در میان نامشان دارند؛ معماری برتر شبکه عصبی کانولوشن را انتخاب می کنند.  معماری ها شامل: 1-معماری ساده 2-معماری ساده عمیق 3-معماری ResNext با عمق های مختلف بررسی می شود.از هر سیگنال بلند مدت، تپش های قلب استخراج و کلاس بندی و برچسب گذاری انجام می شود و برچسب و کلاسی که در قطعات سیگنال بلند مدت بیشتر تکرار شده باشد برچسب نهایی است. دقت کلاس بندی هر کلاس هم از نتایج این کد دریافت و ثبت می شود تا در مرحله 4 برای شاخص بتا استفاده شود.

2-کدی که عبارت 1CNN در میان نامش دارد، داده های بسیاری با 7 روش داده افزایی هر کدام با شدت های مختلف تولید و در پوشه AugEvl ذخیره می کند. در این کد 5 شاخص کیفی پراکندگی برای هر نمونه حاصل از داده افزایی، محاسبه و به انتهای سیگنال آن نمونه اضافه می شود. این شاخص ها از آخر به اول 1-مرزی بودن 2-غیر مرزی بودن 3-نسبت فاصله با همکلاسی ها به فاصله با غیر همکلاسی ها 4-واریانس فاصله با 5 همسایه نزدیک 5-فاصله با همکلاسی ها است.

3-کدی که عبارت 2CNN یا 3CNN در میان نامش دارد، پراکندگی مناسب برای انتخاب نمونه های مصنوعی لازم را تعیین می کند. همان هایی که در پوشه AugEvl ذخیره شده بودند.  نمونه های مصنوعی هر بار بر اساس یکی از 5 شاخص کیفی مرتب می شوند و با یکی از 5 توزیع 1-ترتیبی 2-خطی 3-بتا 4-نمایی 5-یکسان انتخاب می شوند و برای آموزش شبکه عصبی بکار می روند. یعنی 25 حالت انتخاب برای نمونه های حاصل از هر روش داده افزایی بررسی می شود. البته در این کد 3 حالت کیفی-توزیعی از قبل در نظر گرفته شده است و شاخص کیفی-توزیع برتر برای انتخاب نمونه های هر روش داده افزایی تعیین می شود. در این مرحله بهترین دقت کلاس بندی شبکه عصبی آموزش دیده با داده های حاصل از هر روش داده افزایی نیز ثبت می شود. 

4-کدی که عبارت 4CNN در میان نامش دارد، حجم های مختلف داده افزایی را با ضرایب ایکس برای حجم کلی داده افزایی و  آلفا برای حجم هر روش داده افزایی و بتا برای هر کلاس انجام می دهد تا روش داده افزایی برتر بدست آید. در این مرحله روش های Cubuk و alphatrim نیز بررسی شده اند.

5-پوشه هایی که "-" در آخرنام خود دارند نشان می دهد که نتایج به دست آمده با روش ارائه شده بنده، نسبت به تحقیقات اخیر، ضعیف تر  بوده اند.

6-تفاوت تصادفی و مرزی در کد و عنوان فایل ها اشاره به روش انتخاب نمونه پایه برای اعمال روش داده افزایی روی آن دارد. در حالت مرزی شانس بیشتری به نمونه های مرزی و نزدیک به کلاس های مخالف برای داده افزایی لحاظ می شود که طبق مقاله جیدیو است . اما چون در روش حاضر نمونه ها کیفیت سنجی می شوند حجم عظیم داده از همه نمونه ها با تعداد یکسان تولید و کیفیت سنجی و با پراکندگی بهتر انتخاب می شوند..
