awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds00.csv

awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds01.csv

awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds02.csv
awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds03.csv
awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds04.csv

awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds05.csv
awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds06.csv

awk -v l=10000 '(NR==1){header=$0;next}
                (NR%l==2) {
                   close(file); 
                   file=sprintf("%s.%0.5d.csv",FILENAME,++c)
                   sub(/csv[.]/,"",file)
                   print header > file
                }
                {print > file}' ds07.csv
