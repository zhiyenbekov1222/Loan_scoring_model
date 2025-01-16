'''
This project is for Predicting the defaults of the business process "Payment Loan". The model is trained on 
LightGBM 4.5.0. gradient boosting algorithm.

Owner: Zhalgas Zhieyenbekov
zhalgas.zhiyenbekov@tele2.kz
Date last training:10.10.2024
'''

import pandas as pd
import numpy as np
import os
import pickle
import oracledb
from datetime import datetime
from typing import Tuple
from sqlalchemy.dialects.oracle import FLOAT, NUMBER, DATE, VARCHAR2
from sqlalchemy import create_engine
import time
import yaml

model_lightgbm = pickle.load(open("zz_dp_model_0324.pkl", "rb"))
if model_lightgbm:
        print('Model successfully imported.')
class dp_model:
        '''
        This objects calculates the score and transforms the table with available offers for Payment Loan
        '''
        def __init__(self, model_name = 'zz_dp_model_0324.pkl', table_source = 'zzz_dp_pilot_sample_24' ,actual_table = 'cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL', hist_table = 'cbm_project.ZZZ_DP_LOAN_SCORING_NEW_HIST'):
                '''
                Initialization function of class DP model with prerequisites.
                '''
                self.__model_name = model_name
                self.__actual_table = actual_table
                self.__hist_table = hist_table
                self.__table_source = table_source

                print('Scoring DP started successfully.')

        def load_config(self, file_path='oracle_connection.yaml'):
            with open(file_path, 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            return config

        def create_oracle_connection(self, config_path='oracle_connection.yaml'):
            config = self.load_config(config_path)
            dsn = f"{config['host']}:{config['port']}/?service_name={config['service_name']}"
            connection = oracledb.connect(
                user=config['login'],
                password=config['password'],
                dsn=dsn
            )
            return connection

        def get_data_mfs(self) -> pd.DataFrame:
                '''
                This function returns the table with  predictors for scoring.
                '''
                SQL = f'''
                        
                  select * from samaster.DAILY_PREDICTORS_DP_24_ACTUAL

            
                '''
                db = self.create_oracle_connection()
                data = pd.read_sql(SQL, con=db)
                db.close()
                return data

        def get_pilot_data_mfs(self) -> pd.DataFrame:
                '''
                This function returns the table with  predictors for scoring.
                '''
                SQL = f'''
                        SELECT * 
                        FROM {self.__table_source}
                '''
                db = self.create_oracle_connection()
                data = pd.read_sql(SQL, con=db)
                db.close()
                return data

        def get_data_mfs2(self) -> pd.DataFrame:
                '''
                This function returns the table with  predictors for scoring.
                '''
                SQL = f'''
                        SELECT t2.* 
                        FROM zzz_dp_pilot_sample_2_24 t1
                        LEFT JOIN samaster.daily_predictors_dp_24_actual t2 ON t1.subs_id = t2.subs_id
                        where t2.SUBS_ID is not null
                '''
                db = self.create_oracle_connection()
                data = pd.read_sql(SQL, con=db)
                db.close()
                return data
        
        def dwh_ins(self, data= None):
                '''
                Function updates and inserts actual scores in tables 
                '''
                conn_cx = self.create_oracle_connection()
                #cursor = cx.Cursor(conn_cx)
                cursor = conn_cx.cursor()
                df = data

                rowss = df[['DT', 'SUBS_ID', 'AVAILABLE_SERV', 'MODEL_ID', 'SCORES', 'FLAG_STAFF', 'SUBS_LIFETIME', 'ARPU_30D', 'FLAG_WAS_DF', 'RISK_GRADE', 'FLAG_FILTER', 'DTIME_INSERTED']].to_records(index=False).tolist()

                cursor.prepare(''' insert into cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL(DT, SUBS_ID, AVAILABLE_SERV, MODEL_ID, SCORE, FLAG_STAFF, SUBS_LIFETIME, ARPU_30D, FLAG_WAS_DF, RISK_GRADE, FLAG_FILTER, DTIME_INSERTED) values ( TO_DATE(:1, 'YYYY-MM-DD'), :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, TO_DATE(:12, 'YYYY-MM-DD HH24:MI:SS') )''')
                cursor.executemany(None, rowss)
                conn_cx.commit()

                cursor.prepare(''' insert into cbm_project.ZZZ_DP_LOAN_SCORING_NEW_HIST(DT, SUBS_ID, AVAILABLE_SERV, MODEL_ID, SCORE, FLAG_STAFF, SUBS_LIFETIME, ARPU_30D, FLAG_WAS_DF, RISK_GRADE, FLAG_FILTER , DTIME_INSERTED) values ( TO_DATE(:1, 'YYYY-MM-DD'), :2, :3, :4, :5, :6, :7, :8, :9, :10, :11 , TO_DATE(:12, 'YYYY-MM-DD HH24:MI:SS') )''')
                cursor.executemany(None, rowss)
                conn_cx.commit()

                sql_grant = '''GRANT SELECT ON cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL TO PUBLIC'''
                cursor.execute(sql_grant)  
                conn_cx.commit()

                sql_count = '''select max(DT) max_dt, count(*) cnt_all  from cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL'''
                data_report = pd.read_sql(sql_count, con=conn_cx)

                cursor.close()
                conn_cx.close()

                report_date = data_report['MAX_DT'].astype('str')[0]
                report_count = data_report['CNT_ALL'].astype('str')[0]
                print(f"DP Scoring for {report_date} has success {report_count} records.")

                return report_date, report_count

        def dwh_ins_pilot(self, data= None):
                '''
                Function updates and inserts actual scores in Pilot tables 
                '''
                conn_cx = self.create_oracle_connection()
                cursor = conn_cx.cursor()
                df = data

                rowss = df[['DT', 'SUBS_ID', 'AVAILABLE_SERV', 'MODEL_ID', 'SCORES', 'FLAG_STAFF', 'SUBS_LIFETIME', 'ARPU_30D', 'FLAG_WAS_DF', 'RISK_GRADE', 'FLAG_FILTER', 'DTIME_INSERTED']].to_records(index=False).tolist()

                cursor.prepare(''' insert into cbm_project.ZZZ_DP_LOAN_SCORING_NEW_HIST_ALL(DT, SUBS_ID, AVAILABLE_SERV, MODEL_ID, SCORE, FLAG_STAFF, SUBS_LIFETIME, ARPU_30D, FLAG_WAS_DF, RISK_GRADE, FLAG_FILTER, DTIME_INSERTED) values ( TO_DATE(:1, 'YYYY-MM-DD'), :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, TO_DATE(:12, 'YYYY-MM-DD HH24:MI:SS') )''')
                cursor.executemany(None, rowss)
                conn_cx.commit()

                sql_count = '''select max(DT) max_dt, count(*) cnt_all  from cbm_project.ZZZ_DP_LOAN_SCORING_NEW_HIST_ALL'''
                data_report = pd.read_sql(sql_count, con=conn_cx)

                cursor.close()
                conn_cx.close()

                report_date = data_report['MAX_DT'].astype('str')[0]
                report_count = data_report['CNT_ALL'].astype('str')[0]
                print(f"DP Scoring for LAST_PILOT {report_date} has success {report_count} records.")

                return report_date, report_count

        def dwh_truncate(self):
                '''
                Function updates and inserts actual scores in tables
                '''
                conn_cx = self.create_oracle_connection()
                cursor = conn_cx.cursor()  # Fixed cursor creation, `conn_cx.cursor()`

                sql_truncate = ''' 
                        TRUNCATE TABLE cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL
                ''' 
                # Execute the batch insert
                cursor.execute(sql_truncate)
                conn_cx.commit()

                cursor.close()
                conn_cx.close()

                print(f"DP Scoring has successfully updated records.")

        def dwh_hist_scores(self):
                '''
                Function inserts hist scores in tables
                '''
                conn_cx = self.create_oracle_connection()
                cursor = conn_cx.cursor()  # Fixed cursor creation, `conn_cx.cursor()`

                sql_truncate = ''' 
                
                        INSERT INTO CBM_PROJECT.ZZZ_LOAN_SCORING_OLD_HIST_OWN
                        SELECT * FROM loan_scoring
                        
                ''' 
                # Execute the batch insert
                cursor.execute(sql_truncate)
                conn_cx.commit()

                cursor.close()
                conn_cx.close()

                print(f"OLD DP Scoring has successfully inserted to as own records.")

        def dwh_test_insert(self, subs_id, available):
                '''
                Function updates and inserts actual scores in tables
                '''
                conn_cx = self.create_oracle_connection()
                cursor = conn_cx.cursor()  # Fixed cursor creation, `conn_cx.cursor()`

                sql_truncate = f''' 
                        
                        INSERT INTO loan_scoring (DT, SUBS_ID, AVAILABLE_SERV, MODEL_ID, SCORES, SUBS_LIFETIME, ARPU_30D, ANY_ACTIVITY_DAYS_CNT_30D)
                        SELECT TRUNC(SYSDATE), {subs_id}, {available}, 0, 0, 0, 0, 0
                        FROM DUAL
                        WHERE NOT EXISTS (
                            SELECT 1 FROM loan_scoring WHERE SUBS_ID = {subs_id}
                        )

                ''' 
                # Execute the batch insert
                cursor.execute(sql_truncate)
                conn_cx.commit()

                cursor.close()
                conn_cx.close()

                print(f"Test scores are recorded successfully.")

        def dwh_ins_update(self):
                '''
                Function updates and inserts actual scores in tables
                '''
                conn_cx = self.create_oracle_connection()
                cursor = conn_cx.cursor()  # Fixed cursor creation, `conn_cx.cursor()`

                sql_update = ''' 

                        merge into cbm_project.LOAN_SCORING old
                        using (
                        select T1.*, score as scores from cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL T1
                        ) new
                        on (old.SUBS_ID = new.SUBS_ID)
                        when matched then
                        update set 
                                old.AVAILABLE_SERV = new.AVAILABLE_SERV,
                                old.SCORES = new.SCORES

                '''

                cursor.execute(sql_update)
                conn_cx.commit()

                sql_grant = '''GRANT SELECT ON cbm_project.LOAN_SCORING TO PUBLIC'''
                cursor.execute(sql_grant)  
                conn_cx.commit()

                cursor.close()
                conn_cx.close()

                print(f"DP Scoring has successfully updated records.")

        def check_actual_data(self, data):
                '''
                The function checks whether the datamart is updated.
                '''
                conn_cx = self.create_oracle_connection()
                sql_select = ''' select max(DT) AS dt from cbm_project.ZZZ_DP_LOAN_SCORING_NEW_ACTUAL '''
                pd_score = pd.read_sql(sql_select, con=conn_cx)
                date_score = pd_score['DT'][0]
                date_dm = data[['INSERTED_DATETIME']].max()[0]

                print(date_score)
                print(str(date_score))

                if str(date_score) == None:
                        result = date_dm < date_score
                else:
                        result = False
                return result

        def run(self):
                '''
                The main function that executes all procedures-1.
                '''

                print('Function Run successfully called!')
                print('Reading the datamart..')

                data = self.get_data_mfs()


                data[model_lightgbm.feature_name()] = data[model_lightgbm.feature_name()].fillna(0)
                data['SCORE'] = model_lightgbm.predict(data[model_lightgbm.feature_name()])

                data_s = data[[ 'SUBS_ID', 'MSISDN', 'NEW_LIFETIME', 'ARPU_30D', 'MAX_LOAN_DP_3M', 'FLAG_WAS_DF',  'FLAG_WAS_DF_7', 'FLAG_WAS_DF_30',	'FLAG_WAS_NDF',	'LAST_LOAN_DATE', 'MAX_DPD', 'MAX_VALUE', 'CNT_DP', 'FLAG_DP_EXISTING', 'SCORE', 'INSERTED_DATETIME']]

                if ((data_s.isna().sum() / data_s.shape[0]) > 0.7).max():
                    print('Datamart is not correct!')
                    self.dwh_ins_update()
                    return 0

                data_s['RISK_GRADE'] = np.where(((data_s['SCORE'] < 0.45) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['CNT_DP'] >= 1) &
                                                (data_s['MAX_VALUE'] >= 2500) &
                                                (data_s['NEW_LIFETIME'] >= 720) &
                                                (data_s['ARPU_30D'] >= 3000)), 'A+',

                                        np.where(((data_s['SCORE'] < 0.45) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['CNT_DP'] >= 1) &
                                                (data_s['MAX_VALUE'] >= 2500) &
                                                (data_s['NEW_LIFETIME'] >= 720) &
                                                (data_s['ARPU_30D'] >= 250) ), 'A',   

                                        np.where(((data_s['SCORE'] < 0.45) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['CNT_DP'] >= 1) &
                                                (data_s['NEW_LIFETIME'] >= 720) &
                                                (data_s['ARPU_30D'] >= 100) ), 'A-',

                                        np.where(((data_s['SCORE'] < 0.5) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['NEW_LIFETIME'] >= 360) &
                                                (data_s['ARPU_30D'] >= 2500)), 'B+',

                                        np.where(((data_s['SCORE'] < 0.5) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['NEW_LIFETIME'] >= 360) &
                                                (data_s['ARPU_30D'] >= 1000)), 'B',

                                        np.where(((data_s['SCORE'] < 0.5) & 
                                                (data_s['FLAG_DP_EXISTING'] == 1) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['NEW_LIFETIME'] >= 180) &
                                                (data_s['ARPU_30D'] >= 1000)), 'B-',

                                        np.where(((data_s['SCORE'] < 0.6) & 
                                                (data_s['NEW_LIFETIME'] >= 180) &
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['ARPU_30D'] >= 1000)), 'C+',

                                        np.where(((data_s['SCORE'] < 0.6) & 
                                                (data_s['MAX_DPD'] == 0) &
                                                (data_s['NEW_LIFETIME'] >= 90) &
                                                (data_s['ARPU_30D'] >= 1000)), 'C',

                                        np.where(((data_s['SCORE'] < 0.6) & 
                                                (data_s['NEW_LIFETIME'] >= 90) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['ARPU_30D'] >= 100) ), 'C-',

                                        np.where(((data_s['SCORE'] < 0.65) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] >= 180) & 
                                                (data_s['ARPU_30D'] >= 2000)), 'D+',

                                        np.where(((data_s['SCORE'] < 0.65) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] >= 180) & 
                                                (data_s['ARPU_30D'] >= 1000)), 'D',

                                        np.where(((data_s['SCORE'] < 0.65) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] > 90) &
                                                (data_s['ARPU_30D'] >= 500)), 'D-',

                                        np.where(((data_s['SCORE'] < 0.7) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] >= 180) &
                                                (data_s['ARPU_30D'] >= 500)),  'E+',

                                        np.where(((data_s['SCORE'] < 0.7) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] >= 120) &
                                                (data_s['ARPU_30D'] >= 100)),  'E', 

                                        np.where(((data_s['SCORE'] < 0.7) & 
                                                (data_s['MAX_DPD'] == 0) & 
                                                (data_s['NEW_LIFETIME'] >= 90) & 
                                                (data_s['ARPU_30D']  >= 100)),  'E-', 

                                        np.where(((data_s['SCORE'] < 0.45) & 
                                                (data_s['NEW_LIFETIME'] >= 360) & 
                                                (data_s['ARPU_30D']  >= 100)),  'R+', 

                                        np.where(((data_s['SCORE'] < 0.6) & 
                                                (data_s['NEW_LIFETIME'] >= 180) & 
                                                (data_s['ARPU_30D']  >= 0)),  'R', 

                                        np.where(((data_s['SCORE'] < 0.7) & 
                                                (data_s['MAX_DPD'] <= 0) & 
                                                (data_s['NEW_LIFETIME'] >= 120) & 
                                                (data_s['ARPU_30D']  >= 0)),  'R-', 

                                        'F'))))))))))))))))))

                print('RISK A+ : ' , data_s[data_s['RISK_GRADE'].isin(['A+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['A+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK A : '  , data_s[data_s['RISK_GRADE'].isin(['A'])].shape[0], '--> '   , round(data_s[data_s['RISK_GRADE'].isin(['A'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK A- : ' , data_s[data_s['RISK_GRADE'].isin(['A-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['A-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK B+ : ' , data_s[data_s['RISK_GRADE'].isin(['B+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['B+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK B : '  , data_s[data_s['RISK_GRADE'].isin(['B'])].shape[0], '--> '   , round(data_s[data_s['RISK_GRADE'].isin(['B'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK B- : ' , data_s[data_s['RISK_GRADE'].isin(['B-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['B-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK C+ : ' , data_s[data_s['RISK_GRADE'].isin(['C+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['C+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK C : '  , data_s[data_s['RISK_GRADE'].isin(['C'])].shape[0], '--> '   , round(data_s[data_s['RISK_GRADE'].isin(['C'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK C- : ' , data_s[data_s['RISK_GRADE'].isin(['C-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['C-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK D+ : ' , data_s[data_s['RISK_GRADE'].isin(['D+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['D+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK D : '  , data_s[data_s['RISK_GRADE'].isin(['D'])].shape[0], '--> '   , round(data_s[data_s['RISK_GRADE'].isin(['D'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK D- : ' , data_s[data_s['RISK_GRADE'].isin(['D-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['D-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK E+ : ' , data_s[data_s['RISK_GRADE'].isin(['E+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['E+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK E : '  , data_s[data_s['RISK_GRADE'].isin(['E'])].shape[0], '--> '   , round(data_s[data_s['RISK_GRADE'].isin(['E'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK E- : ' , data_s[data_s['RISK_GRADE'].isin(['E-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['E-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK R+ : ' , data_s[data_s['RISK_GRADE'].isin(['R+'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['R+'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK R : ' , data_s[data_s['RISK_GRADE'].isin(['R'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['R'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000
                print('RISK R- : ' , data_s[data_s['RISK_GRADE'].isin(['R-'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['R-'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                print('RISK F : ' , data_s[data_s['RISK_GRADE'].isin(['F'])].shape[0], '--> '  , round(data_s[data_s['RISK_GRADE'].isin(['F'])].shape[0]   / data_s.shape[0]*100, 2), '%')    # 10 000

                data_s['FLAG_STAFF'] = 0
                data_s['MODEL_ID'] = 0
                data_s['ANY_ACTIVITY_DAYS_CNT_30D'] = 30
                data_s['FLAG_FILTER'] = 1
                data_s['SCORES'] = data_s['SCORE']

                data_s['DT'] = datetime.now().strftime('%Y-%m-%d')
                data_s['DTIME_INSERTED'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                data_s['FLAG_WAS_DF'] = data_s['FLAG_WAS_DF'].fillna(0)
                data_s = data_s.rename(columns = {'NEW_LIFETIME': 'SUBS_LIFETIME'})

                data_s['AVAILABLE_SERV'] = np.where((data_s['RISK_GRADE'].isin(['A+', 'A', 'A-', 'B+', 'B', 'B-'])), 4,
                                        np.where((data_s['RISK_GRADE'].isin(['C+','C','C-'])), 3,
                                        np.where((data_s['RISK_GRADE'].isin(['D+','D', 'D-', 'E+', 'E', 'R+'])), 2,
                                        np.where((data_s['RISK_GRADE'].isin(['E-', 'R', 'R-'])), 1, 0))))

                dct_hand_subs = {70162741: 2, 8421896: 1, 34014615:4, 70822334: 1, 14956305:1, 75875970: 4,
                        65214576: 4, 70427809: 4, 69185728: 4, 69873858: 3, 69721211: 3, 67822069: 3, 
                        74161798: 1, 74064982: 4, 74219663: 1, 76202858: 4, 75303988: 4, 75368151: 4, 
                        75188483: 2, 67677676: 4, 75002839: 3, 73134286: 4, 65214568: 4, 76816025: 1, 
                        76816017: 2, 76595370: 3, 76595489: 4, 77227501: 4, 76364551: 1, 28785069: 3, 
                        74992391: 2, 77869351: 4, 77719430: 3, 77677793: 1, 78436193: 3, 78121937: 2, 
                        77542144: 4, 78333858: 1, 76341708: 1}

                data_s['AVAILABLE_SERV'] = data_s['SUBS_ID'].map(dct_hand_subs).combine_first(data_s['AVAILABLE_SERV'])
                data_s['FLAG_WAS_DF'] = data_s['FLAG_WAS_DF'].fillna(0)
                data_s[['AVAILABLE_SERV', 'FLAG_WAS_DF']] = data_s[['AVAILABLE_SERV', 'FLAG_WAS_DF']].astype(int)

                print(data_s.shape)

                self.dwh_hist_scores()

                data_pilot_1 = self.get_pilot_data_mfs()
                subs_pilot = data_pilot_1['SUBS_ID'].values
                data_s_polot = data_s[data_s['SUBS_ID'].isin(subs_pilot)]

                self.dwh_truncate()
                self.dwh_ins(data = data_s_polot)
                self.dwh_ins_update()

                for i in range(0,data_s.shape[0], 1000000):
                    print(data_s[i:i+1000000].shape)
                    self.dwh_ins_pilot(data = data_s[i:i+1000000])
                print('DP Finished successfully!')

dp = dp_model()    
dp.run()